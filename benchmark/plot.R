library(ggplot2)
library(reshape2)
library(scales)
library(RColorBrewer)
library(dplyr)
library(tidyr)
library(cowplot)


# Constants
# #########################################
# #########################################
# Config
TIMEOUT <- 1*60

# Names
py_compiler_names <- c("numba")
c_compiler_names <- c("clang", "polly", "polly-parallel", "polly-vectorizer", "polly-parallel-vectorizer")

tensor_compiler_cpu_names <- c("jax", "numpy", "torch", "torch_compiled")
tensor_compiler_gpu_names <- c("jax_gpu_onlykernel")

# Colors
cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

cTensorize <- "#ff7f00"
cTenspiler <- "#e41a1c"
cMlirSynth <- "#377eb8"
cMlt <- "#4daf4a"
cPolly <- "#dd72ee"
cClang <- "#000000"

cNumpy <- "#999999"
cJax <- "#E69F00"
cTorch <- "#009E73"
cTorchCompiled <- "#56B4E9"

cGrey <- "#3b3b3b"


# Helper functions 
# #########################################
# #########################################
gm_mean <- function(x, na.rm=TRUE){
  exp(sum(log(x[x > 0]), na.rm=na.rm) / length(x))
}

load_df_all <- function() {
  df1 <- read.csv("data/amd-7950x.csv")
  df1$platform <- "amd"
  
  df2 <- read.csv("data/intel-8700k.csv")
  df2$platform <- "intel"
  
  df <- rbind(df1, df2)
  
  # Flatten from time1, time2, time3, ... to run, time
  df <- melt(df, id.vars=c("benchmark", "size", "category", "variant", "platform"), variable.name="run", value.name="time")
  # Remove the run column
  df <- subset(df, select = -c(run))
  # Rename all "polybench" prefixed categories to "polybench"
  df$category <- gsub("^polybench.*", "polybench", df$category)
  
  return(df)
}

rename_methods <- function(df) {
  orig <- c("clang", "polly", "mlt", "tenspiler", "mlirSynth", "tensorize")
  changed <- c("LLVM-O3", "Polly", "MultiLevelTactics", "Tenspiler", "MlirSynth", "Tensorize")
  df$method <- factor(df$method, levels=orig)
  df$method <- factor(df$method, levels=orig, labels <- changed)
  
  return(df)
}

rename_cpu_platforms <- function(df) {
  df$platform <- factor(df$platform, levels=c("amd", "intel"))
  df$platform <- factor(df$platform, levels=c("amd", "intel"),
                        labels=c("AMD Ryzen 9 7950X", "Intel I7-8700K"))
  return(df)
}

rename_gpu_platforms <- function(df) {
  df$platform <- factor(df$platform, levels=c("amd", "intel"))
  df$platform <- factor(df$platform, levels=c("amd", "intel"),
                        labels=c("NVIDIA GTX 1080Ti (AMD platform)", "NVIDIA GTX 1080Ti (on Intel)"))
  return(df)
}

rename_compilers <- function(df) {
  df$variant <- factor(df$variant, levels=c("jax", "numpy", "torch", "torch_compiled", "jax_gpu", "jax_gpu_onlykernel", "torch_gpu", "torch_gpu_onlykernel"))
  df$variant <- factor(df$variant, levels=c("jax", "numpy", "torch", "torch_compiled", "jax_gpu", "jax_gpu_onlykernel", "torch_gpu", "torch_gpu_onlykernel"),
                      labels=c("JAX", "NumPy", "PyTorch", "PyTorch (Compiled)", "JAX", "JAX", "PyTorch", "PyTorch"))
  return(df)
}

get_best_compiler <- function(df_in, compiler_names) {
  df <- subset(df_in, variant %in% compiler_names)
  df <- df %>% group_by(benchmark, category, platform) %>% top_n(-1, time)
  
  return(df)
}

get_speedups <- function(df_in, df_baseline, compiler_names) {
  # Calculate speedups of `compiler_names` compilers against c_best
  # - Select the tensor cpu compiler df
  df <- subset(df_in, variant %in% compiler_names)
  # - Compute speedups
  df <- merge(df, df_baseline, by=c("benchmark"))
  df$speedup <- df$time.y / df$time.x
  # - Clean dfframe
  df <- subset(df, select = c(benchmark, category.x, variant.x, platform.x, speedup))
  names(df)[names(df) == "variant.x"] <- "variant"
  names(df)[names(df) == "category.x"] <- "category"
  names(df)[names(df) == "platform.x"] <- "platform"
  
  return(df)
}

get_speedups_over_baseline <- function(df_in, compiler_names) {
  # Get best C and Py compilers
  c_best <- get_best_compiler(df_in, c("clang"))
  py_best <- get_best_compiler(df_in, c("numba"))
  # Calculate speedups
  c_speedups <- get_speedups(df_in, c_best, compiler_names)
  c_speedups$compilers <- "LLVM-O3, Polly, GCC"
  py_speedups <- get_speedups(df_in, py_best, compiler_names)
  py_speedups$compilers <- "Numba"
  
  df <- rbind(c_speedups, py_speedups)
  
  # Per (benchmark, category), keep only the variants that achieve the best speedup
  df <- df %>% group_by(benchmark, category, compilers) %>% top_n(1, speedup)
  
  return(df)
}

get_synth_times <- function() {
  # Load tensorize data
  df_tensorize <- read.csv("out/stats.csv")
  df_tensorize <- subset(df_tensorize, select = c("benchmark_suite", "benchmark", "synthesis_time"))
  names(df_tensorize)[names(df_tensorize) == "benchmark_suite"] <- "category"
  df_tensorize$method <- "tensorize"

  # Load baseline data
  df <- read.csv("data/synth_times_baselines.csv")

  # Merge the two dataframes
  df <- rbind(df, df_tensorize)

  return(df)
}

# Plots 
# #########################################
# #########################################

# Plot 1: Speedup overview
# #########################################
get_synthesis_speedups <- function(df_in, timeout) {
  # Load synth time data
  df <- get_synth_times()

  df <- df %>% complete(method, nesting(benchmark, category), fill=list(synthesis_time=3600))
  
  df$lifted <- df$synthesis_time < timeout
  
  df <- merge(df_in, df, by=c("benchmark", "category"))
  
  # Assign all rows that have "lifted" FALSE a speedup of 1
  df$speedup[!df$lifted] <- 1
  
  return(df)
}

get_speedups_over_baseline <- function(df, baseline) {
  clang <- subset(df, variant == baseline)
  df <- merge(df, clang, by=c("benchmark", "category", "platform"))
  df$speedup <- df$time.y / df$time.x
  # Clean dataframe
  df <- subset(df, select = c(benchmark, category, variant.x, speedup, platform))
  names(df)[names(df) == "variant.x"] <- "variant"
  
  return(df)
}

get_best_polly_speedups <- function(df_all, compiler_names) {
  df <- get_speedups_over_baseline(df_all, "clang")
  df <- subset(df, variant %in% c("polly", "polly-parallel", "polly-vectorizer", "polly-parallel-vectorizer"))
  df <- df %>% group_by(benchmark, category, platform) %>% top_n(1, speedup)
  df$method <- "polly"
  return(df)
}

get_best_synth_speedups <- function(df_all, baseline, compiler_names, timeout) {
  df <- get_best_compiler(df_all, baseline)
  df <- get_speedups(df_all, df, compiler_names)
  df <- df%>% group_by(benchmark, category, platform) %>% top_n(1, speedup)
  df <- get_synthesis_speedups(df, timeout)
  return(df)
}

get_gm_speedups <- function(df_all, compiler_names, timeout) {
  # CPU
  # - Clang
  df_polly <- get_best_polly_speedups(df_all)
  df_c_synth <- get_best_synth_speedups(df_all, "clang", tensor_compiler_cpu_names, timeout)
  df_c_gm <- rbind(df_polly, df_c_synth)
  df_c_gm <- aggregate(speedup ~ method + platform, df_c_gm, gm_mean)
  df_c_gm$baseline <- "C (LLVM-O3)"
  
  # - Numba
  df_py_synth <- get_best_synth_speedups(df_all, "numba", tensor_compiler_cpu_names, timeout)
  df_py_gm <- aggregate(speedup ~ method + platform, df_py_synth, gm_mean)
  df_py_gm$baseline <- "Python (Numba)"
  
  df_cpu <- rbind(df_c_gm, df_py_gm)
  df_cpu$device <- "CPU"
  
  # GPU
  # - Clang
  df_c_synth <- get_best_synth_speedups(df_all, "clang", tensor_compiler_gpu_names, timeout)
  df_c_gm <- aggregate(speedup ~ method + platform, df_c_synth, gm_mean)
  df_c_gm$baseline <- "C (LLVM-O3)"

  # - Numba
  df_py_synth <- get_best_synth_speedups(df_all, "numba", tensor_compiler_gpu_names, timeout)
  df_py_gm <- aggregate(speedup ~ method + platform, df_py_synth, gm_mean)
  df_py_gm$baseline <- "Python (Numba)"

  df_gpu <- rbind(df_c_gm, df_py_gm)
  df_gpu$device <- "GPU"
  
  df <- rbind(df_cpu, df_gpu)
  
  return(df)
}

df_all <- load_df_all()
df_gm <- get_gm_speedups(df_all, c_compiler_names, TIMEOUT)
#df_gm <- subset(df_gm, platform == "amd")
df_gm <- subset(df_gm, baseline == "C (LLVM-O3)")
for (platform in c("amd", "intel")) {
  for (device in c("CPU", "GPU")) {
    df_gm <- rbind(df_gm, data.frame(method="clang", platform=platform, speedup=1, baseline="C (LLVM-O3)", device=device))
  }
}

df_gm$method <- factor(df_gm$method, levels=c("clang", "polly", "mlt", "tenspiler", "mlirSynth", "tensorize"))
df_gm <- rename_methods(df_gm)

plot_overview <- function(df_plt, ylim_y, my_colors, order) {
  # Reorder platforms
  df_plt$platform <- factor(df_plt$platform, levels=order)
  
  plt <- ggplot(df_plt, aes(x=method, y=speedup, fill=method)) +
    geom_bar(stat="identity", position=position_dodge(width=0.9), width=0.85) +
    facet_grid(~.+platform, scales="free_x", space = "free") +
    #facet_wrap(~device+platform, scales="free_y", nrow = 1) +
    scale_y_continuous(
      trans = pseudo_log_trans(base = 10),
      breaks = c(1, 10^(0:8)),
      minor_breaks = rep(1:9, 21)*(10^rep(-10:10, each=9)),
      labels = label_number(accuracy = 1)
    ) +
    labs(x="Method", y="Geomean speedup (log)") +
    coord_cartesian(ylim=c(1, ylim_y)) +
    geom_hline(yintercept=1, linetype="dashed") +
    geom_text(aes(label=sprintf("%.1fx", speedup)), vjust=-0.5, size=3, position=position_dodge(width=1)) +
    theme_minimal() +
    theme(legend.position="none") +
    scale_fill_manual(values=my_colors) +
    #scale_fill_manual(values=cbPalette) +
    theme(axis.text.x = element_text(angle = 15, hjust = 1))
  
  return(plt)
}
p1 <- plot_overview(subset(rename_cpu_platforms(df_gm), device == "CPU"), 13, c(cClang, cPolly, cMlt, cTenspiler, cMlirSynth, cTensorize), c("Intel I7-8700K", "AMD Ryzen 9 7950X"))
p2 <- plot_overview(subset(rename_gpu_platforms(df_gm), platform == "NVIDIA GTX 1080Ti (AMD platform)" & device == "GPU"), 6000, c(cClang, cMlt, cTenspiler, cMlirSynth, cTensorize), c("NVIDIA GTX 1080Ti (AMD platform)"))
cowplot::plot_grid(p1, p2, nrow=1, rel_widths = c(2, 1))

ggsave("out/figure_7.pdf", width=10, height=3, units="in")

# Plot 2: Coverage of benchmark suites #
#########################################
df <- get_synth_times()
df <- df %>% complete(method, nesting(benchmark, category), fill=list(synthesis_time=3600))
df$lifted <- df$synthesis_time < TIMEOUT
df_cov <- df %>% group_by(method, category) %>% summarize(coverage=sum(lifted)/n() * 100)

# Add a total row for each method
df_totals <- df %>% group_by(method) %>% summarize(coverage=sum(lifted)/n() * 100)
df_totals$category <- "Total"

df <- rbind(df_cov, df_totals)

# Reorder
# - Methods
df$method <- factor(df$method, levels=c("mlt", "tenspiler", "mlirSynth", "tensorize"))
df <- rename_methods(df)
# - Categories so that "Total" is last
cats <- unique(df$category)
cats <- c(cats[cats != "Total"], "Total")
df$category <- factor(df$category, levels=cats)

bench <- ggplot(subset(df, category != "Total"), aes(x=category, y=coverage, fill=method)) +
  geom_bar(stat="identity", position=position_dodge(width=0.9), width=0.8) +
  #theme(plot.margin=margin(5.5, 0, 5.5, 5.5)) +
  labs(x="Benchmark Suite", y="Success rate in %") +
  theme_minimal() +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 5)) +
  coord_cartesian(clip = "off") +
  scale_fill_manual(values=c(cMlt, cTenspiler, cMlirSynth, cTensorize), guide="none")

agg <- ggplot(subset(df, category == "Total"), aes(x=category, y=coverage, fill=method)) +
  geom_bar(stat="identity", position=position_dodge(width=0.9), width=0.8) +
  labs(x=" ", y=NULL) + # whitespace keeps padding
  #theme(axis.text.y=element_blank(), axis.ticks.y = element_blank(), plot.margin=margin(5.5, 5.5, 5.5, 0)) +
  theme_minimal() +
  theme(axis.text.y=element_blank(), axis.ticks.y = element_blank()) +
  guides(fill=guide_legend(title="Method", reverse=T)) +
  geom_vline(xintercept=11.5) +
  scale_y_continuous(limits = c(0, 100), expand = c(0, 5)) +
  coord_cartesian(clip = "off") +
  geom_text(data=subset(df, category == "Total"), aes(label=sprintf("%.f%%", coverage)), vjust=-0.5, size=3, position=position_dodge(width=.9)) +
  scale_fill_manual(values=c(cMlt, cTenspiler, cMlirSynth, cTensorize))

cowplot::plot_grid(bench, agg, nrow=1, rel_widths=c(3.1, 1))

ggsave("out/figure_8.pdf", width=10, height=2, units="in")

# Plot 3: Synthesis time on x axis, success rate on y axis. Different colors for different methods.
# #########################################
df <- get_synth_times()

df <- df %>% complete(method, nesting(benchmark, category), fill=list(synthesis_time=3600))

# Create a empty dataframe with columns "synthesis_time", "method", "num_successes"
df_succ <- data.frame()

# Iterate from 0 to 3600
for (i in 1:3599) {
  # Count the number of successes for each method
  df_tmp <- df %>% filter(synthesis_time <= i) %>% group_by(method) %>% summarize(num_successes=n())
  df_tmp$synthesis_time <- i
  df_succ <- rbind(df_succ, df_tmp)
}

# Calculate the success rate. Total number of benchmarks is 99
df_succ$num_successes <- df_succ$num_successes / 99 * 100

df_succ <- rename_methods(df_succ)
df_succ$method <- factor(df_succ$method, levels=c("Tensorize", "MlirSynth", "Tenspiler", "MultiLevelTactics"))

plot_3 <- function(df_succ) {
  plt <- ggplot(df_succ, aes(x=synthesis_time, y=num_successes, color=method)) +
    geom_line(size=1) +
    scale_x_continuous(
      trans = pseudo_log_trans(base = 10),
      breaks = c(1, 10, 100, 1000, 3600),
      #breaks = c(1, 10, 100, 3600),
      minor_breaks = rep(1:9, 21)*(10^rep(-10:10, each=9)),
      labels = label_number(accuracy = 1),
      expand = c(0, 0), limits = c(1, 4000)
    ) +
    coord_cartesian(ylim=c(0, 100), clip="off") +
    geom_vline(xintercept = 3600, linetype="dashed") +
    labs(x="Synthesis time in s (log)", y="Success rate in %", color="Method") +
    scale_colour_manual(values=c(cTensorize, cMlirSynth, cTenspiler, cMlt)) +
    theme_minimal()
  return(plt)
}

plt <- plot_3(df_succ)
print(plt)
ggsave("out/figure_9.pdf", width=5, height=2, units="in")

# Plot 5: Complexities
# #########################################
# Load data
df_compl <- read.csv("data/complexities.csv")
# Rename all "polybench" prefixed categories to "polybench"
df_compl$category <- gsub("^polybench.*", "polybench", df_compl$category)

df_synth_times <- get_synth_times()
df_synth_times <- df_synth_times %>% complete(method, nesting(benchmark, category), fill=list(synthesis_time=3600))
# Merge data
df <- merge(df_compl, df_synth_times, by=c("benchmark", "category"))

# Filter out "mlt"
df <- subset(df, method != "mlt")
df <- subset(df, select = -c(max_arg_rank, num_tensor_contractions))

# Flatten from time1, time2, time3, ... to run, time
df <- melt(df, id.vars=c("benchmark", "category", "method", "synthesis_time"), variable.name="complexity_metric", value.name="value")

# Rename
df <- rename_methods(df)
df$method <- factor(df$method, levels=c("Tensorize", "MlirSynth", "Tenspiler"))

df$complexity_metric <- gsub("max_loop_depth", "Max Loop Depth (Source)", df$complexity_metric)
df$complexity_metric <- gsub("num_high_level_ops", "Program Length (Tensor DSL)", df$complexity_metric)

df$complexity_metric <- factor(df$complexity_metric, levels=c("Max Loop Depth (Source)", "Program Length (Tensor DSL)"))

# Plot a facet_wrap of complexity_metric. Each facet contains boxplots of the synthesis time for each method.
plot_5 <- function(df, x_lim, boxplot_width) {
  plt <- ggplot(df, aes(x=value, y=synthesis_time, colour=method)) +
    #geom_boxplot(aes(x=factor(value)), position=position_dodge(width=boxplot_width), width=boxplot_width) +
    geom_smooth(span = 1.2, level=0.5) +
    facet_wrap(~complexity_metric, scales="free", ncol=1) +
    labs(y="Synthesis time in s", x="Complexity", fill="Method", color="Method") +
    coord_cartesian(ylim=c(0, 3800), xlim=x_lim) +
    scale_colour_manual(values=c(cTensorize, cMlirSynth, cTenspiler)) +
    scale_x_continuous(breaks = function(x) unique(floor(pretty(seq(min(x), (max(x) + 1) * 1.1))))) +
    theme_minimal()
  return(plt)
} 
p1 = plot_5(subset(df, complexity_metric == "Max Loop Depth (Source)"), c(1, 4), 0.4)
p2 = plot_5(subset(df, complexity_metric == "Program Length (Tensor DSL)"), c(1, 12), 0.75)

prow <- plot_grid(
  p1 + theme(legend.position="none"),
  p2 + theme(legend.position="none"),
  nrow = 2
)
plot_grid(prow, get_legend(p1), ncol = 2, rel_widths = c(1, 0.2))
ggsave("out/figure_10.pdf", width=5, height=4, units="in")