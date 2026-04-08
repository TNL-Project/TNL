import random
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

CUDA_COLORS = {"CUDA": "#FF6B6B", "Host": "#4ECDC4"}


def create_top_bar_chart(df, value_col, title, color_scale, x_label):
    top_df = df.nlargest(20, value_col)
    fig = px.bar(
        top_df,
        x=value_col,
        y="target",
        orientation="h",
        color=value_col,
        color_continuous_scale=color_scale,
        labels={value_col: x_label, "target": "Target"},
        title=title,
    )
    fig.update_layout(height=600, yaxis={"categoryorder": "total ascending"})
    return fig


def create_scatter_plot(df, x_col, y_col, title, x_label, y_label, has_cuda_label):
    kwargs = {
        "x": x_col,
        "y": y_col,
        "hover_data": ["target"],
        "labels": {x_col: x_label, y_col: y_label},
        "title": title,
    }
    if has_cuda_label:
        kwargs["color"] = "target_type"
        kwargs["color_discrete_map"] = CUDA_COLORS
    return px.scatter(df, **kwargs)


def create_histogram(df, value_col, title, x_label, nbins=50):
    return px.histogram(
        df, x=value_col, nbins=nbins, labels={value_col: x_label}, title=title
    )


def categorize_target(target):
    if "tnl-benchmark-" in target or "Benchmark" in target:
        return "Benchmarks"
    elif "Test" in target or "_test" in target.lower():
        return "Tests"
    elif "Example" in target:
        return "Examples"
    elif target.startswith("tnl-"):
        return "Tools"
    else:
        return "Other"


def simulate_memory_peak(task_data, jobs, strategy):
    tasks = list(task_data)
    if strategy == "slowest":
        tasks.sort(key=lambda x: x[1], reverse=True)
    elif strategy == "fastest":
        tasks.sort(key=lambda x: x[1])
    workers_time = [0.0] * jobs
    workers_mem = [0.0] * jobs
    max_memory = 0.0

    for task_mem, task_time in tasks:
        min_worker = workers_time.index(min(workers_time))
        workers_time[min_worker] += task_time
        workers_mem[min_worker] = task_mem
        current_total_mem = sum(workers_mem)
        if current_total_mem > max_memory:
            max_memory = current_total_mem

    return max_memory


def simulate_memory_average(task_data, jobs, iterations=100):
    total = 0
    for _ in range(iterations):
        shuffled = list(task_data)
        random.shuffle(shuffled)
        workers_time = [0.0] * jobs
        workers_mem = [0.0] * jobs
        max_memory = 0.0

        for task_mem, task_time in shuffled:
            min_worker = workers_time.index(min(workers_time))
            workers_time[min_worker] += task_time
            workers_mem[min_worker] = task_mem
            current_total_mem = sum(workers_mem)
            if current_total_mem > max_memory:
                max_memory = current_total_mem

        total += max_memory
    return total / iterations


def find_safe_jobs_memory(task_data, memory_limit_mb):
    for j in range(1, 33):
        peak = simulate_memory_peak(task_data, j, "slowest")
        if peak > memory_limit_mb:
            return j - 1
    return 32


def simulate_scheduler(task_times, jobs, strategy):
    tasks = list(task_times)
    if strategy == "slowest":
        tasks.sort(reverse=True)
    elif strategy == "fastest":
        tasks.sort()
    workers = [0.0] * jobs
    for task in tasks:
        min_worker = workers.index(min(workers))
        workers[min_worker] += task
    return max(workers)


def simulate_random_average(task_times, jobs, iterations=100):
    total = 0
    for _ in range(iterations):
        shuffled = list(task_times)
        random.shuffle(shuffled)
        workers = [0.0] * jobs
        for task in shuffled:
            min_worker = workers.index(min(workers))
            workers[min_worker] += task
        total += max(workers)
    return total / iterations


st.set_page_config(
    page_title="Memory Analysis Dashboard", page_icon="📊", layout="wide"
)

st.title("Build Time and Memory Usage Analysis Dashboard")

st.markdown(
    "This dashboard visualizes build metrics collected by "
    "`scripts/analyze-build-targets.sh`, which builds each target individually using "
    "`/usr/bin/time --verbose just build <target>` and records peak memory usage "
    "(Maximum resident set size), elapsed build time, binary size, host function "
    "count, and CUDA kernel count."
)

with st.sidebar:
    st.header("Dataset Selection")
    results_dir = Path("build/targets_analysis")
    csv_files = sorted(results_dir.glob("targets_summary_*.csv"), reverse=True)

    if not csv_files:
        st.error(
            "No analysis results found. Run scripts/analyze-build-targets.sh first."
        )
        st.stop()

    selected_file = st.selectbox(
        "Select analysis run:",
        csv_files,
        format_func=lambda x: x.stem.replace("targets_summary_", ""),
    )

df = pd.read_csv(selected_file)

if df.empty:
    st.warning("No data in selected file.")
    st.stop()

df["max_rss_mb"] = df["max_rss_kb"] / 1024

has_binary_size = "binary_size_kb" in df.columns
if has_binary_size:
    df["binary_size_mb"] = df["binary_size_kb"] / 1024

has_cuda_label = "is_cuda" in df.columns
if has_cuda_label:
    df["target_type"] = df["is_cuda"].apply(lambda x: "CUDA" if x else "Host")

has_host_functions = "host_functions" in df.columns
has_cuda_kernels = "cuda_kernels" in df.columns

with st.sidebar:
    st.header("Overview")
    binary_line = (
        f"- **Total Binary Size:** {df['binary_size_mb'].sum():.0f} MiB  \n"
        if has_binary_size
        else ""
    )
    st.markdown(
        f"- **Total Targets:** {len(df)}  \n"
        f"- **Max Memory:** {df['max_rss_mb'].max():.1f} MiB  \n"
        f"- **Avg Memory:** {df['max_rss_mb'].mean():.1f} MiB  \n"
        f"- **Total Sequential Time:** {df['elapsed_time_sec'].sum() / 60:.0f} min  \n"
        f"- **Max Time:** {df['elapsed_time_sec'].max():.1f} s  \n"
        f"- **Avg Time:** {df['elapsed_time_sec'].mean():.1f} s  \n"
        f"{binary_line}"
    )

tab_memory, tab_time, tab_binary, tab_corr = st.tabs(
    ["Memory Usage", "Build Time", "Binary Insights", "Correlations"]
)

with tab_memory:
    st.subheader("Top 20 Targets by Memory Usage")
    fig_mem_bar = create_top_bar_chart(
        df, "max_rss_mb", "Top 20 Memory Consumers", "Reds", "Memory (MiB)"
    )
    st.plotly_chart(fig_mem_bar)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Memory Distribution")
        fig_hist = create_histogram(
            df, "max_rss_mb", "Distribution of Memory Usage", "Memory (MiB)"
        )
        st.plotly_chart(fig_hist)

    with col_right:
        st.subheader("Memory vs Build Time")
        fig_scatter = create_scatter_plot(
            df,
            "elapsed_time_sec",
            "max_rss_mb",
            "Memory vs Build Time",
            "Build Time (s)",
            "Memory (MiB)",
            has_cuda_label,
        )
        st.plotly_chart(fig_scatter)

    st.subheader("Memory Usage TreeMap")

    df_treemap = df.copy()
    df_treemap["category"] = df_treemap["target"].apply(categorize_target)
    df_treemap["short_target"] = df_treemap["target"].str.replace("tnl-benchmark-", "")
    df_treemap["short_target"] = df_treemap["short_target"].str.replace("tnl-", "")

    fig_treemap = px.treemap(
        df_treemap,
        path=["category", "short_target"],
        values="max_rss_mb",
        color="max_rss_mb",
        color_continuous_scale="RdYlGn_r",
        title="Memory Usage by Target (click to explore)",
        labels={"max_rss_mb": "Memory (MiB)"},
    )
    st.plotly_chart(fig_treemap)

    st.subheader("Parallel Build Memory Estimation")
    st.info(
        "Simulates memory usage during parallel builds using different scheduling "
        "strategies. Shows the maximum memory used at any point during the build."
    )

    task_data = list(zip(df["max_rss_mb"].values, df["elapsed_time_sec"].values))
    sorted_by_time_desc = sorted(task_data, key=lambda x: x[1], reverse=True)
    sorted_by_time_asc = sorted(task_data, key=lambda x: x[1])

    max_single_target_mb = df["max_rss_mb"].max()
    max_target_name = df.loc[df["max_rss_mb"].idxmax(), "target"]

    safe_8 = find_safe_jobs_memory(task_data, 8192)
    safe_16 = find_safe_jobs_memory(task_data, 16384)
    safe_32 = find_safe_jobs_memory(task_data, 32768)

    jobs_range = list(range(1, 33))
    slowest_first_mems_mb = [
        simulate_memory_peak(sorted_by_time_desc, j, "slowest") for j in jobs_range
    ]
    fastest_first_mems_mb = [
        simulate_memory_peak(sorted_by_time_asc, j, "fastest") for j in jobs_range
    ]
    random_mems_mb = [simulate_memory_average(task_data, j) for j in jobs_range]

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            "Largest Target", f"{max_single_target_mb:.0f} MiB", f"{max_target_name}"
        )

        mem_jobs = st.slider(
            "Parallel Jobs", min_value=1, max_value=32, value=4, key="mem_jobs"
        )

        slowest_first_mem = slowest_first_mems_mb[mem_jobs - 1]
        fastest_first_mem = fastest_first_mems_mb[mem_jobs - 1]
        random_mem = random_mems_mb[mem_jobs - 1]

        st.metric(
            "Slowest First (LPT)",
            f"{slowest_first_mem:.0f} MiB",
            f"({slowest_first_mem / 1024:.1f} GiB)",
        )
        st.metric(
            "Random Order",
            f"{random_mem:.0f} MiB",
            f"({random_mem / 1024:.1f} GiB)",
        )
        st.metric(
            "Fastest First (SPT)",
            f"{fastest_first_mem:.0f} MiB",
            f"({fastest_first_mem / 1024:.1f} GiB)",
        )

    with col2:
        fig_mem = go.Figure()
        fig_mem.add_trace(
            go.Scatter(
                x=jobs_range,
                y=[m / 1024 for m in slowest_first_mems_mb],
                mode="lines+markers",
                name="Slowest First (LPT)",
                hovertemplate="%{y:.1f} GiB at %{x} jobs<extra>Slowest First</extra>",
            )
        )
        fig_mem.add_trace(
            go.Scatter(
                x=jobs_range,
                y=[m / 1024 for m in random_mems_mb],
                mode="lines+markers",
                name="Random Order",
                hovertemplate="%{y:.1f} GiB at %{x} jobs<extra>Random</extra>",
            )
        )
        fig_mem.add_trace(
            go.Scatter(
                x=jobs_range,
                y=[m / 1024 for m in fastest_first_mems_mb],
                mode="lines+markers",
                name="Fastest First (SPT)",
                hovertemplate="%{y:.1f} GiB at %{x} jobs<extra>Fastest First</extra>",
            )
        )

        fig_mem.add_hline(
            y=8, line_dash="dash", line_color="orange", annotation_text="8 GiB"
        )
        fig_mem.add_hline(
            y=16, line_dash="dash", line_color="red", annotation_text="16 GiB"
        )
        fig_mem.add_hline(
            y=32, line_dash="dash", line_color="darkred", annotation_text="32 GiB"
        )

        fig_mem.update_layout(
            title="Peak Memory vs Parallel Jobs (Scheduling Strategies)",
            xaxis_title="Number of Parallel Jobs",
            yaxis_title="Peak Memory (GiB)",
            height=400,
            showlegend=True,
        )

        st.plotly_chart(fig_mem)

    st.info(
        "**Slowest First (LPT):** Longest targets run first, memory-heavy targets "
        "may overlap.\n\n"
        "**Fastest First (SPT):** Short targets finish quickly, reducing memory "
        "overlap.\n\n"
        "**Random:** Average case when build system has no particular ordering."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Safe with 8 GiB RAM", f"{safe_8} jobs")
    with col_b:
        st.metric("Safe with 16 GiB RAM", f"{safe_16} jobs")
    with col_c:
        st.metric("Safe with 32 GiB RAM", f"{safe_32} jobs")

with tab_time:
    st.subheader("Top 20 Targets by Build Time")
    df["elapsed_time_min"] = df["elapsed_time_sec"] / 60
    fig_time_bar = create_top_bar_chart(
        df, "elapsed_time_min", "Top 20 Slowest Targets", "Greens", "Build Time (min)"
    )
    st.plotly_chart(fig_time_bar)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Build Time Distribution")
        fig_hist_time = create_histogram(
            df, "elapsed_time_sec", "Distribution of Build Times", "Build Time (s)"
        )
        st.plotly_chart(fig_hist_time)

    with col_right:
        st.subheader("Memory vs Build Time")
        fig_scatter_time = create_scatter_plot(
            df,
            "elapsed_time_sec",
            "max_rss_mb",
            "Memory vs Build Time",
            "Build Time (s)",
            "Memory (MiB)",
            has_cuda_label,
        )
        st.plotly_chart(fig_scatter_time, key="scatter_time")

    st.subheader("Build Time TreeMap")
    df_treemap_time = df.copy()
    df_treemap_time["category"] = df_treemap_time["target"].apply(categorize_target)
    df_treemap_time["short_target"] = df_treemap_time["target"].str.replace(
        "tnl-benchmark-", ""
    )
    df_treemap_time["short_target"] = df_treemap_time["short_target"].str.replace(
        "tnl-", ""
    )
    fig_treemap_time = px.treemap(
        df_treemap_time,
        path=["category", "short_target"],
        values="elapsed_time_sec",
        color="elapsed_time_sec",
        color_continuous_scale="Greens",
        title="Build Time by Target (click to explore)",
        labels={"elapsed_time_sec": "Build Time (s)"},
    )
    st.plotly_chart(fig_treemap_time)

    st.subheader("Build Time Estimation")

    total_sequential_time = df["elapsed_time_sec"].sum()
    longest_target_time = df["elapsed_time_sec"].max()
    sorted_times_desc = df["elapsed_time_sec"].sort_values(ascending=False).values
    sorted_times_asc = df["elapsed_time_sec"].sort_values().values
    all_times = df["elapsed_time_sec"].values

    jobs_range_time = list(range(1, 33))
    slowest_first_times_sec = [
        simulate_scheduler(sorted_times_desc, j, "slowest") for j in jobs_range_time
    ]
    fastest_first_times_sec = [
        simulate_scheduler(sorted_times_asc, j, "fastest") for j in jobs_range_time
    ]
    random_times_sec = [simulate_random_average(all_times, j) for j in jobs_range_time]
    ideal_times = [total_sequential_time / 60 / j for j in jobs_range_time]

    col_t1, col_t2 = st.columns([1, 2])

    with col_t1:
        st.metric("Total Sequential Time", f"{total_sequential_time / 60:.0f} min")

        time_jobs = st.slider(
            "Parallel Jobs", min_value=1, max_value=32, value=4, key="time_jobs"
        )

        slowest_time = slowest_first_times_sec[time_jobs - 1]
        fastest_time = fastest_first_times_sec[time_jobs - 1]
        random_time = random_times_sec[time_jobs - 1]

        st.metric(
            "Slowest First (LPT)",
            f"{slowest_time / 60:.1f} min",
            f"{total_sequential_time / slowest_time:.1f}x faster",
        )
        st.metric(
            "Random Order",
            f"{random_time / 60:.1f} min",
            f"{total_sequential_time / random_time:.1f}x faster",
        )
        st.metric(
            "Fastest First (SPT)",
            f"{fastest_time / 60:.1f} min",
            f"{total_sequential_time / fastest_time:.1f}x faster",
        )

    with col_t2:
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Scatter(
                x=jobs_range_time,
                y=[t / 60 for t in slowest_first_times_sec],
                mode="lines+markers",
                name="Slowest First (LPT)",
                hovertemplate="%{y:.1f} min at %{x} jobs<extra>Slowest First</extra>",
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=jobs_range_time,
                y=[t / 60 for t in random_times_sec],
                mode="lines+markers",
                name="Random Order",
                hovertemplate="%{y:.1f} min at %{x} jobs<extra>Random</extra>",
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=jobs_range_time,
                y=[t / 60 for t in fastest_first_times_sec],
                mode="lines+markers",
                name="Fastest First (SPT)",
                hovertemplate="%{y:.1f} min at %{x} jobs<extra>Fastest First</extra>",
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=jobs_range_time,
                y=ideal_times,
                mode="lines",
                name="Ideal (Linear)",
                line=dict(dash="dot"),
                hovertemplate="%{y:.1f} min at %{x} jobs<extra>Ideal</extra>",
            )
        )
        fig_time.add_hline(
            y=longest_target_time / 60,
            line_dash="dash",
            line_color="red",
            annotation_text="Min possible",
        )

        fig_time.update_layout(
            title="Build Time vs Parallel Jobs (Scheduling Strategies)",
            xaxis_title="Number of Parallel Jobs",
            yaxis_title="Build Time (min)",
            height=400,
        )

        st.plotly_chart(fig_time)

    st.info(
        "**Slowest First (LPT):** Best practical strategy - processes longest targets "
        "first, minimizing idle time at the end.\n\n"
        "**Fastest First (SPT):** Worst strategy - short tasks finish quickly leaving "
        "few workers for long tasks.\n\n"
        "**Random:** Average case when build system has no particular ordering."
    )

with tab_binary:
    st.subheader("Binary Size Analysis")

    if has_binary_size:
        col_bin1, col_bin2 = st.columns(2)

        with col_bin1:
            fig_binary_bar = create_top_bar_chart(
                df,
                "binary_size_mb",
                "Top 20 Targets by Binary Size",
                "Blues",
                "Binary Size (MiB)",
            )
            st.plotly_chart(fig_binary_bar)

        with col_bin2:
            fig_mem_bin = create_scatter_plot(
                df,
                "binary_size_mb",
                "max_rss_mb",
                "Compile-Time Memory vs Binary Size",
                "Binary Size (MiB)",
                "Compile Memory (MiB)",
                has_cuda_label,
            )
            st.plotly_chart(fig_mem_bin)
    else:
        st.info("Binary size data not available in this dataset.")

    if has_host_functions or has_cuda_kernels:
        st.markdown("---")
        st.subheader("Function & Kernel Analysis")

        cuda_targets = None
        if has_cuda_kernels and (df["cuda_kernels"] > 0).any():
            cuda_targets = df[df["cuda_kernels"] > 0].sort_values(
                "cuda_kernels", ascending=False
            )[:20]

        col_func1, col_func2 = st.columns(2)

        with col_func1:
            if has_host_functions:
                fig_funcs_bar = create_top_bar_chart(
                    df,
                    "host_functions",
                    "Top 20 Targets by Host Functions",
                    "Purples",
                    "Host Functions",
                )
                st.plotly_chart(fig_funcs_bar)

        with col_func2:
            if cuda_targets is not None:
                fig_kern_bar = create_top_bar_chart(
                    cuda_targets,
                    "cuda_kernels",
                    "Top 20 CUDA Targets by Kernels",
                    "Oranges",
                    "CUDA Kernels",
                )
                st.plotly_chart(fig_kern_bar)

        col_scatter1, col_scatter2 = st.columns(2)

        with col_scatter1:
            if has_host_functions:
                fig_func_mem = create_scatter_plot(
                    df,
                    "host_functions",
                    "max_rss_mb",
                    "Host Functions vs Compile Memory",
                    "Host Functions",
                    "Compile Memory (MiB)",
                    has_cuda_label,
                )
                st.plotly_chart(fig_func_mem)

        with col_scatter2:
            if cuda_targets is not None:
                fig_kernel_mem = create_scatter_plot(
                    df[df["cuda_kernels"] > 0],
                    "cuda_kernels",
                    "max_rss_mb",
                    "CUDA Kernels vs Compile Memory",
                    "CUDA Kernels",
                    "Compile Memory (MiB)",
                    has_cuda_label,
                )
                st.plotly_chart(fig_kernel_mem)

with tab_corr:
    st.subheader("Correlation Matrix")

    corr_cols = ["max_rss_mb", "elapsed_time_sec"]
    if has_binary_size:
        corr_cols.append("binary_size_mb")
    if has_host_functions:
        corr_cols.append("host_functions")
    if has_cuda_kernels:
        corr_cols.append("cuda_kernels")

    corr_labels = {
        "max_rss_mb": "Memory (MiB)",
        "elapsed_time_sec": "Time (s)",
        "binary_size_mb": "Binary (MiB)",
        "host_functions": "Host Funcs",
        "cuda_kernels": "CUDA Kernels",
    }

    col_corr1, col_corr2 = st.columns(2)

    with col_corr1:
        st.markdown("**Host Targets**")
        df_host = df[~df["is_cuda"]] if has_cuda_label else df
        if len(df_host) > 1:
            corr_host = df_host[corr_cols].corr()
            corr_host_labeled = corr_host.rename(index=corr_labels, columns=corr_labels)
            fig_corr_host = px.imshow(
                corr_host_labeled,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
                title=f"Host Targets (n={len(df_host)})",
            )
            fig_corr_host.update_layout(height=500)
            st.plotly_chart(fig_corr_host)
        else:
            st.info("Not enough host targets for correlation analysis.")

    with col_corr2:
        st.markdown("**CUDA Targets**")
        df_cuda = df[df["is_cuda"]] if has_cuda_label else pd.DataFrame()
        if len(df_cuda) > 1:
            corr_cuda = df_cuda[corr_cols].corr()
            corr_cuda_labeled = corr_cuda.rename(index=corr_labels, columns=corr_labels)
            fig_corr_cuda = px.imshow(
                corr_cuda_labeled,
                labels=dict(color="Correlation"),
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
                title=f"CUDA Targets (n={len(df_cuda)})",
            )
            fig_corr_cuda.update_layout(height=500)
            st.plotly_chart(fig_corr_cuda)
        else:
            st.info("Not enough CUDA targets for correlation analysis.")

    st.markdown("---")
    st.subheader("Custom Scatter Plot")

    col_x, col_y = st.columns(2)

    with col_x:
        x_axis = st.selectbox(
            "X-Axis",
            corr_cols,
            index=0,
            format_func=lambda x: corr_labels.get(x, x),
        )

    with col_y:
        y_axis = st.selectbox(
            "Y-Axis",
            corr_cols,
            index=1,
            format_func=lambda x: corr_labels.get(x, x),
        )

    fig_custom_scatter = create_scatter_plot(
        df,
        x_axis,
        y_axis,
        f"{corr_labels.get(y_axis, y_axis)} vs {corr_labels.get(x_axis, x_axis)}",
        corr_labels.get(x_axis, x_axis),
        corr_labels.get(y_axis, y_axis),
        has_cuda_label,
    )
    st.plotly_chart(fig_custom_scatter, key="custom_scatter")

st.markdown("---")
st.subheader("All Results")

filter_pattern = st.text_input("Filter targets (regex):", value="")
if filter_pattern:
    filtered_df = df[df["target"].str.contains(filter_pattern, case=False, regex=True)]
else:
    filtered_df = df

display_cols = ["target", "max_rss_mb", "elapsed_time_sec"]
if has_binary_size:
    display_cols.append("binary_size_mb")
if "host_functions" in filtered_df.columns:
    display_cols.append("host_functions")
if "cuda_kernels" in filtered_df.columns:
    display_cols.append("cuda_kernels")

display_df = filtered_df[display_cols].sort_values("max_rss_mb", ascending=False)

column_config = {
    "target": st.column_config.TextColumn("Target"),
    "max_rss_mb": st.column_config.NumberColumn("Memory (MiB)", format="%.1f"),
    "elapsed_time_sec": st.column_config.NumberColumn("Time (s)", format="%.2f"),
}
if has_binary_size:
    column_config["binary_size_mb"] = st.column_config.NumberColumn(
        "Binary (MiB)", format="%.1f"
    )
if "host_functions" in filtered_df.columns:
    column_config["host_functions"] = st.column_config.NumberColumn(
        "Host Funcs", format="%d"
    )
if "cuda_kernels" in filtered_df.columns:
    column_config["cuda_kernels"] = st.column_config.NumberColumn(
        "CUDA Kernels", format="%d"
    )

st.dataframe(
    display_df,
    column_config=column_config,
    height=400,
)

st.download_button(
    "Download CSV",
    filtered_df.to_csv(index=False).encode(),
    "memory_analysis.csv",
    "text/csv",
)
