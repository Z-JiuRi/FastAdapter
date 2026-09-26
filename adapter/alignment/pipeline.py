from .dependencies import *  # noqa: F401,F403
from .io import *  # noqa: F401,F403
from .features import *  # noqa: F401,F403
from .matching import *  # noqa: F401,F403
from .reporting import *  # noqa: F401,F403
from .cli import *  # noqa: F401,F403

def main() -> None:
    args = parse_args()
    weight_ratio, activation_ratio = validate_args(args)
    args.weight_ratio = weight_ratio
    args.activation_ratio = activation_ratio
    args.task_root = resolve_task_root(args.data_root, args.dataset, args.scenario)
    if not args.task_root.is_dir():
        raise RuntimeError(f"adapter task root does not exist: {args.task_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_stream = (args.output_dir / "analysis.log").open(
        "w", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_stream)
    torch.set_num_threads(args.torch_num_threads)
    gpu_pool = parse_gpu_pool(args)
    if gpu_pool is not None:
        gpu_devices, gpu_tasks = gpu_pool
        worker_devices = [
            device
            for device, capacity in zip(gpu_devices, gpu_tasks)
            for _ in range(capacity)
        ]
        aggregation_device = torch.device("cpu")
        device_label = ",".join(
            f"{device}:{capacity}"
            for device, capacity in zip(gpu_devices, gpu_tasks)
        )
    else:
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {device}")
        worker_devices = [device] * args.workers
        aggregation_device = device
        device_label = str(device)
    worker_count = len(worker_devices)

    limits = {
        "train": args.limit_train,
        "val": args.limit_val,
        "test": args.limit_test,
    }
    discovered_records = {
        split: discover_tasks(split, limits[split], args.task_root, args.adapter_name)
        for split in args.splits
    }
    records = {
        split: split_records
        for split, split_records in discovered_records.items()
        if split_records
    }
    if "train" not in records:
        raise RuntimeError(f"no compact train Adapters found under {args.task_root}")
    for split, split_records in discovered_records.items():
        if not split_records:
            print(f"SKIP split={split} reason=no compact Adapters under {args.task_root / split}")

    reference_record = records["train"][0]
    if args.reference_task is not None:
        matches = [record for record in records["train"] if record.name == args.reference_task]
        if not matches:
            raise ValueError(
                f"--reference-task={args.reference_task!r} is not in selected train tasks"
            )
        reference_record = matches[0]

    reference_cpu_state = load_state(reference_record)
    reference_device_state = to_device_state(reference_cpu_state, aggregation_device)
    raw_reference_signatures = (
        weight_signatures(reference_cpu_state, aggregation_device, args.bias_scale)
        if args.cost_type in {"weight", "hybrid"}
        else None
    )
    activation_indices = None
    raw_reference_activations = None
    reference_train_rows = 0
    if args.cost_type in {"activation", "hybrid"}:
        reference_train_rows, _ = load_code_shape(reference_record)
        activation_indices = shared_activation_indices(
            reference_train_rows,
            args.activation_samples,
            args.activation_sample_seed,
        )
        reference_codes = load_indexed_codes(
            reference_record,
            activation_indices,
            aggregation_device,
        )
        raw_reference_activations = post_gelu_activations(
            reference_device_state,
            reference_codes,
        )
        del reference_codes

    print(
        "READ_ONLY_ALIGNMENT "
        f"task_root={args.task_root} devices={device_label} cost_type={args.cost_type} "
        f"ratios=({weight_ratio:.3f},{activation_ratio:.3f}) "
        f"bias_scale={args.bias_scale:.6g} "
        f"tasks={','.join(f'{split}:{len(split_records)}' for split, split_records in records.items())} "
        f"reference={reference_record.name} "
        f"iterations={args.iterations} train_rows="
        f"{reference_train_rows} activation_rows="
        f"{0 if activation_indices is None else len(activation_indices)} "
        f"activation_sample_seed={args.activation_sample_seed} "
        f"feature_cache={args.feature_cache} "
        f"adapter_name={args.adapter_name} save_aligned={int(args.save_aligned)} "
        f"aligned_name={args.aligned_name} "
        f"workers={worker_count} torch_num_threads={args.torch_num_threads} "
        f"output_dir={args.output_dir.resolve()}"
    )

    unique_worker_devices = list(dict.fromkeys(worker_devices))
    raw_reference_cache: dict[str, tuple[
        OrderedDict[str, torch.Tensor],
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]] = {}
    for worker_device in unique_worker_devices:
        raw_reference_cache[str(worker_device)] = (
            to_device_state(reference_cpu_state, worker_device),
            [value.to(worker_device) for value in raw_reference_signatures]
            if raw_reference_signatures is not None
            else None,
            [value.to(worker_device) for value in raw_reference_activations]
            if raw_reference_activations is not None
            else None,
        )

    device_slots: Queue[torch.device] = Queue()
    for worker_device in worker_devices:
        device_slots.put(worker_device)

    def run_in_device_slot(function: Any, *function_args: Any) -> Any:
        worker_device = device_slots.get()
        try:
            if worker_device.type == "cuda":
                with torch.cuda.device(worker_device):
                    return function(*function_args, worker_device)
            return function(*function_args, worker_device)
        finally:
            device_slots.put(worker_device)

    def record_key(record: TaskRecord) -> str:
        return f"{record.split}/{record.name}"

    sampled_activation_rows = 0 if activation_indices is None else len(activation_indices)
    use_feature_cache = (
        args.feature_cache == "on"
        or (
            args.feature_cache == "auto"
            and (
                args.cost_type == "weight"
                or 0 < sampled_activation_rows <= 16384
            )
        )
    )
    args.feature_cache_enabled = use_feature_cache
    feature_cache: dict[str, FeatureBundle] = {}

    def compute_cached_features(
        record: TaskRecord,
        worker_device: torch.device,
    ) -> tuple[str, FeatureBundle]:
        state, signatures, activations = task_features(
            record, worker_device, args.cost_type, activation_indices, args.bias_scale
        )
        bundle = FeatureBundle(
            state=OrderedDict(
                (key, value.detach().cpu().contiguous())
                for key, value in state.items()
            ),
            signatures=(
                [value.detach().cpu().contiguous() for value in signatures]
                if signatures is not None
                else None
            ),
            activations=(
                [value.detach().cpu().contiguous() for value in activations]
                if activations is not None
                else None
            ),
        )
        return record_key(record), bundle

    if use_feature_cache:
        all_records = [
            record for split_records in records.values() for record in split_records
        ]
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="adapter-cache",
            initializer=initialize_worker,
            initargs=(args.torch_num_threads,),
        ) as cache_executor:
            futures = {
                cache_executor.submit(
                    run_in_device_slot,
                    compute_cached_features,
                    record,
                ): record
                for record in all_records
            }
            total_futures = len(futures)
            for index, future in enumerate(as_completed(futures), start=1):
                key, bundle = future.result()
                feature_cache[key] = bundle
                futures.pop(future)
                if index % args.progress_every == 0 or index == total_futures:
                    print(
                        f"FEATURE_CACHE processed={index}/{total_futures} "
                        f"activation_rows={sampled_activation_rows}"
                    )
        if record_key(reference_record) in feature_cache:
            reference_bundle = feature_cache[record_key(reference_record)]
            reference_cpu_state = reference_bundle.state
            raw_reference_signatures = reference_bundle.signatures
            raw_reference_activations = reference_bundle.activations
        print(f"FEATURE_CACHE_READY tasks={len(feature_cache)} enabled=1")
    else:
        print(
            "FEATURE_CACHE_READY tasks=0 enabled=0 "
            f"activation_rows={sampled_activation_rows}"
        )

    def get_raw_features(record: TaskRecord, worker_device: torch.device) -> tuple[
        OrderedDict[str, torch.Tensor],
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        cached = feature_cache.get(record_key(record))
        if cached is not None:
            return (
                to_device_state(cached.state, worker_device),
                [value.to(worker_device, non_blocking=True) for value in cached.signatures]
                if cached.signatures is not None
                else None,
                [value.to(worker_device, non_blocking=True) for value in cached.activations]
                if cached.activations is not None
                else None,
            )
        if record == reference_record:
            return raw_reference_cache[str(worker_device)]
        return task_features(
            record, worker_device, args.cost_type, activation_indices, args.bias_scale
        )

    def align_for_barycenter(
        record: TaskRecord,
        signature_references: dict[str, list[torch.Tensor] | None],
        activation_references: dict[str, list[torch.Tensor] | None],
        worker_device: torch.device,
    ) -> tuple[list[torch.Tensor] | None, list[torch.Tensor] | None, float]:
        _, signatures, activations = get_raw_features(record, worker_device)
        current_signatures = signature_references[str(worker_device)]
        current_activations = activation_references[str(worker_device)]
        orders, block_metrics = align_task_features(
            signatures,
            current_signatures,
            activations,
            current_activations,
            weight_ratio,
            activation_ratio,
        )
        aligned_signatures = (
            [
                unit_rows(signatures[block])[orders[block].to(worker_device)].to(
                    aggregation_device
                )
                for block in range(4)
            ]
            if signatures is not None
            else None
        )
        aligned_activations = (
            [
                unit_columns(activations[block])[
                    :, orders[block].to(worker_device)
                ].to(aggregation_device)
                for block in range(4)
            ]
            if activations is not None
            else None
        )
        return (
            aligned_signatures,
            aligned_activations,
            mean_metric(block_metrics, "matched_cost"),
        )

    def aggregate_barycenter_chunk(
        chunk: list[TaskRecord],
        signature_references: dict[str, list[torch.Tensor] | None],
        activation_references: dict[str, list[torch.Tensor] | None],
        worker_device: torch.device,
    ) -> tuple[
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
        list[float],
        int,
    ]:
        if worker_device.type == "cuda":
            torch.cuda.set_device(worker_device)
        current_signatures = signature_references[str(worker_device)]
        current_activations = activation_references[str(worker_device)]
        signature_sums = (
            [torch.zeros_like(value) for value in current_signatures]
            if current_signatures is not None
            else None
        )
        activation_sums = (
            [torch.zeros_like(value) for value in current_activations]
            if current_activations is not None
            else None
        )
        matched_costs = []
        for record in chunk:
            _, signatures, activations = get_raw_features(record, worker_device)
            orders, block_metrics = align_task_features(
                signatures,
                current_signatures,
                activations,
                current_activations,
                weight_ratio,
                activation_ratio,
            )
            if signature_sums is not None and signatures is not None:
                for block in range(4):
                    signature_sums[block].add_(
                        unit_rows(signatures[block])[orders[block].to(worker_device)]
                    )
            if activation_sums is not None and activations is not None:
                for block in range(4):
                    activation_sums[block].add_(
                        unit_columns(activations[block])[
                            :, orders[block].to(worker_device)
                        ]
                    )
            matched_costs.append(mean_metric(block_metrics, "matched_cost"))
        return (
            [value.to(aggregation_device) for value in signature_sums]
            if signature_sums is not None
            else None,
            [value.to(aggregation_device) for value in activation_sums]
            if activation_sums is not None
            else None,
            matched_costs,
            len(chunk),
        )

    def split_records_by_device(
        current_records: list[TaskRecord],
    ) -> list[tuple[torch.device, list[TaskRecord]]]:
        chunks = [(device, []) for device in unique_worker_devices]
        for index, record in enumerate(current_records):
            chunks[index % len(chunks)][1].append(record)
        return [(device, chunk) for device, chunk in chunks if chunk]

    reference_signatures = raw_reference_signatures
    reference_activations = raw_reference_activations
    barycenter_rows: list[dict[str, float]] = []
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="adapter-align",
        initializer=initialize_worker,
        initargs=(args.torch_num_threads,),
    ) as executor:
        for iteration in range(1, args.iterations + 1):
            signature_references = {
                str(worker_device): (
                    [value.to(worker_device) for value in reference_signatures]
                    if reference_signatures is not None
                    else None
                )
                for worker_device in unique_worker_devices
            }
            activation_references = {
                str(worker_device): (
                    [value.to(worker_device) for value in reference_activations]
                    if reference_activations is not None
                    else None
                )
                for worker_device in unique_worker_devices
            }
            signature_sums = (
                [torch.zeros_like(value) for value in reference_signatures]
                if reference_signatures is not None
                else None
            )
            activation_sums = (
                [torch.zeros_like(value) for value in reference_activations]
                if reference_activations is not None
                else None
            )
            futures = {
                executor.submit(
                    aggregate_barycenter_chunk,
                    chunk,
                    signature_references,
                    activation_references,
                    device,
                ): (device, chunk)
                for device, chunk in split_records_by_device(records["train"])
            }
            total_tasks = len(records["train"])
            processed_tasks = 0
            iteration_costs = []
            for future in as_completed(futures):
                aligned_signatures, aligned_activations, matched_costs, processed = (
                    future.result()
                )
                futures.pop(future)
                processed_tasks += processed
                iteration_costs.extend(matched_costs)
                if signature_sums is not None and aligned_signatures is not None:
                    for block in range(4):
                        signature_sums[block].add_(aligned_signatures[block])
                if activation_sums is not None and aligned_activations is not None:
                    for block in range(4):
                        activation_sums[block].add_(aligned_activations[block])
                print(
                    f"BARYCENTER iteration={iteration}/{args.iterations} "
                    f"processed={processed_tasks}/{total_tasks}"
                )
            if signature_sums is not None:
                reference_signatures = [
                    value / len(records["train"]) for value in signature_sums
                ]
            if activation_sums is not None:
                reference_activations = [
                    value / len(records["train"]) for value in activation_sums
                ]
            print(
                f"BARYCENTER_RESULT iteration={iteration} "
                f"matched_cost_{summarize(iteration_costs)}"
            )
            barycenter_rows.append(
                {"iteration": iteration, **summary_stats(iteration_costs)}
            )
            del signature_references, activation_references

        frozen_hash = reference_hash(reference_signatures, reference_activations)
        print(f"FROZEN_REFERENCE sha256={frozen_hash}")

        probe_generator = torch.Generator().manual_seed(args.probe_seed)
        cpu_probes = torch.randn(
            args.probe_samples, 512, generator=probe_generator
        )
        probes_by_device = {
            str(worker_device): cpu_probes.to(worker_device)
            for worker_device in unique_worker_devices
        }
        frozen_signature_references = {
            str(worker_device): (
                [value.to(worker_device) for value in reference_signatures]
                if reference_signatures is not None
                else None
            )
            for worker_device in unique_worker_devices
        }
        frozen_activation_references = {
            str(worker_device): (
                [value.to(worker_device) for value in reference_activations]
                if reference_activations is not None
                else None
            )
            for worker_device in unique_worker_devices
        }

        def evaluate_task(
            record: TaskRecord,
            worker_device: torch.device,
        ) -> tuple[
            TaskRecord,
            dict[str, float],
            bool,
            list[dict[str, float]],
            list[list[int]],
        ]:
            state, signatures, activations = get_raw_features(record, worker_device)
            orders, block_metrics = align_task_features(
                signatures,
                frozen_signature_references[str(worker_device)],
                activations,
                frozen_activation_references[str(worker_device)],
                weight_ratio,
                activation_ratio,
            )
            aligned = apply_orders(state, orders)
            max_abs, rel_mse = verify_function(
                state, aligned, probes_by_device[str(worker_device)]
            )
            passed = max_abs <= args.function_tolerance
            if args.save_aligned and passed:
                save_aligned_state(aligned, record.directory / args.aligned_name)
            result = {
                "identity_cost": mean_metric(block_metrics, "identity_cost"),
                "matched_cost": mean_metric(block_metrics, "matched_cost"),
                "improvement": mean_metric(block_metrics, "improvement"),
                "changed_ratio": mean_metric(block_metrics, "changed_ratio"),
                "local_margin": mean_metric(block_metrics, "local_margin"),
                "max_abs": max_abs,
                "rel_mse": rel_mse,
            }
            for optional_key in ("matched_weight_cost", "matched_activation_cost"):
                optional_value = mean_metric(block_metrics, optional_key)
                if not np.isnan(optional_value):
                    result[optional_key] = optional_value
            return (
                record,
                result,
                passed,
                block_metrics,
                [order.tolist() for order in orders],
            )

        failed = 0
        all_results: dict[str, list[dict[str, float]]] = {
            split: [] for split in records
        }
        task_rows: list[dict[str, Any]] = []
        block_rows: list[dict[str, Any]] = []
        permutations: dict[str, list[list[int]]] = {}
        for split, split_records in records.items():
            futures = {
                executor.submit(run_in_device_slot, evaluate_task, record): record
                for record in split_records
            }
            total_futures = len(futures)
            completed = []
            for index, future in enumerate(as_completed(futures), start=1):
                completed.append(future.result())
                futures.pop(future)
                if index % args.progress_every == 0 or index == total_futures:
                    print(
                        f"EVALUATION split={split} "
                        f"processed={index}/{total_futures}"
                    )
            for record, result, passed, block_metrics, orders in sorted(
                completed, key=lambda item: item[0].name
            ):
                failed += int(not passed)
                all_results[split].append(result)
                task_rows.append(
                    {
                        "split": split,
                        "name": record.name,
                        "function_check": "PASS" if passed else "FAIL",
                        **result,
                    }
                )
                block_rows.extend(
                    {
                        "split": split,
                        "name": record.name,
                        "block": block,
                        **metrics,
                    }
                    for block, metrics in enumerate(block_metrics)
                )
                permutations[f"{split}/{record.name}"] = orders
                max_abs = result["max_abs"]
                rel_mse = result["rel_mse"]
                component_text = ""
                if "matched_weight_cost" in result:
                    component_text += (
                        f" weight_matched={result['matched_weight_cost']:.6e}"
                    )
                if "matched_activation_cost" in result:
                    component_text += (
                        f" activation_matched={result['matched_activation_cost']:.6e}"
                    )
                print(
                    f"TASK split={split} name={record.name} "
                    f"identity={result['identity_cost']:.6e} "
                    f"matched={result['matched_cost']:.6e} "
                    f"improvement={result['improvement']:.6e} "
                    f"changed={result['changed_ratio']:.4f} "
                    f"margin={result['local_margin']:.6e} "
                    f"components=[{component_text.strip()}] "
                    f"max_abs={max_abs:.6e} rel_mse={rel_mse:.6e} "
                    f"function_check={'PASS' if passed else 'FAIL'}"
                )

            summary_keys = [
                "identity_cost", "matched_cost", "improvement", "changed_ratio",
                "local_margin", "max_abs", "rel_mse",
            ]
            for optional_key in ("matched_weight_cost", "matched_activation_cost"):
                if all(optional_key in result for result in all_results[split]):
                    summary_keys.append(optional_key)
            for key in summary_keys:
                print(
                    f"SUMMARY split={split} metric={key} "
                    f"{summarize([result[key] for result in all_results[split]])}"
                )

    write_analysis_outputs(
        args.output_dir,
        args,
        device_label,
        reference_record,
        frozen_hash,
        barycenter_rows,
        task_rows,
        block_rows,
        permutations,
    )
    print(
        f"FINAL function_checks={sum(len(value) for value in all_results.values())} "
        f"failed={failed} output_dir={args.output_dir.resolve()} "
        f"reference_sha256={frozen_hash}"
    )
    sys.stdout.flush()
    sys.stdout = original_stdout
    log_stream.close()
    if failed:
        raise SystemExit(1)
