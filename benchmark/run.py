import json
import os
import subprocess

import tqdm
from absl import app
from absl import flags
from timeit import default_timer as timer


FLAGS = flags.FLAGS
flags.DEFINE_string("target", "numpy", "Raising target")
flags.DEFINE_string("out_dir", "out", "Result file containing statistics")
flags.DEFINE_integer("n_iter", 1, "Number of iterations")
flags.DEFINE_bool("noguide", False, "Don't select operations automatically")
flags.DEFINE_list("benchmarks", ["polybench", "software"], "Benchmarks to run")
flags.DEFINE_integer("timeout", 30, "Timeout for each benchmark in minutes")


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../tensorize")
SYNTHESIZER_PROGRAM = os.path.join(SCRIPT_DIR, "main.py")

# fmt: off

BENCH_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

POLYBENCH = [
    ("polybench", BENCH_DIR + "polybench/gesummv.mlir"),
    ("polybench", BENCH_DIR + "polybench/2mm.mlir"),
    ("polybench", BENCH_DIR + "polybench/3mm.mlir"),
    ("polybench", BENCH_DIR + "polybench/atax.mlir"),
    ("polybench", BENCH_DIR + "polybench/bicg.mlir"),
    ("polybench", BENCH_DIR + "polybench/covariance.mlir"),
    ("polybench", BENCH_DIR + "polybench/doitgen.mlir"),
    ("polybench", BENCH_DIR + "polybench/gemm.mlir"),
    ("polybench", BENCH_DIR + "polybench/gemver.mlir"),
    ("polybench", BENCH_DIR + "polybench/mvt.mlir"),
    ("polybench", BENCH_DIR + "polybench/correlation.mlir"),
    ("polybench", BENCH_DIR + "polybench/symm.mlir"),
    ("polybench", BENCH_DIR + "polybench/syrk.mlir"),
    ("polybench", BENCH_DIR + "polybench/syr2k.mlir"),
    ("polybench", BENCH_DIR + "polybench/trmm.mlir"),
]

SOFTWARE = [
    # Blend
    ("blend", BENCH_DIR + "software/blend/color_burn_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/color_dodge_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/darken_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/dissolve_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/lighten_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/linear_burn_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/linear_dodge_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/multiply_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/normal_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/normal_blend_f.mlir"),
    ("blend", BENCH_DIR + "software/blend/overlay_blend_8.mlir"),
    ("blend", BENCH_DIR + "software/blend/screen_blend_8.mlir"),

    # Llama
    ("llama", BENCH_DIR + "software/llama/matmul.mlir"),
    ("llama", BENCH_DIR + "software/llama/rmsnorm_part1.mlir"),
    ("llama", BENCH_DIR + "software/llama/rmsnorm_part2.mlir"),
    ("llama", BENCH_DIR + "software/llama/softmax_part1.mlir"),
    ("llama", BENCH_DIR + "software/llama/softmax_part2.mlir"),
    ("llama", BENCH_DIR + "software/llama/softmax_part3.mlir"),
    ("llama", BENCH_DIR + "software/llama/softmax_part4.mlir"),
    #("llama", DIR + "software/llama/transformer_part1.mlir"),
    #("llama", DIR + "software/llama/transformer_part2.mlir"),
    ("llama", BENCH_DIR + "software/llama/transformer_part3.mlir"),
    ("llama", BENCH_DIR + "software/llama/transformer_part4.mlir"),

    # Blas
    ("blas", BENCH_DIR + "software/blas/dot.mlir"),
    ("blas", BENCH_DIR + "software/blas/gemv.mlir"),
    ("blas", BENCH_DIR + "software/blas/ger.mlir"),

    # Darknet
    ("darknet", BENCH_DIR + "software/darknet/gemm_nn.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/gemm_nt.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/gemm_tn.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/gemm_tt.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/mag_array.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/matrix_add_matrix.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/mse_array.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/mult_add_into_cpu.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/ol_l2_cpu1.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/ol_l2_cpu2.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/scale_array.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/scale_matrix.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/sum_array.mlir"),
    ("darknet", BENCH_DIR + "software/darknet/translate_array.mlir"),

    # Dsp
    ("dsp", BENCH_DIR + "software/dsp/matadd.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/matinit.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/matmul.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/matscal.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/matsub.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vadd.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vcopy.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vfill.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vmul.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vneg.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/voffset.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vrecip.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vscal.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/vsub.mlir"),
    ("dsp", BENCH_DIR + "software/dsp/w_vec.mlir"),

    # DspStone
    ("dspstone", BENCH_DIR + "software/dspstone/mat1x3.mlir"),
    ("dspstone", BENCH_DIR + "software/dspstone/matrix1.mlir"),
    ("dspstone", BENCH_DIR + "software/dspstone/matrix2.mlir"),
    ("dspstone", BENCH_DIR + "software/dspstone/n_real_updates.mlir"),
    ("dspstone", BENCH_DIR + "software/dspstone/pin_down.mlir"),

    # Makespeare
    ("makespeare", BENCH_DIR + "software/makespeare/sum_of_squares.mlir"),

    # Mathfu
    ("mathfu", BENCH_DIR + "software/mathfu/diveq_sca.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/diveq.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/len_sq.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/len.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/lerp.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/matmul_sca.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/muleq_sca.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/muleq.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/negate.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/pluseq.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/subeq_sca.mlir"),
    ("mathfu", BENCH_DIR + "software/mathfu/subeq.mlir"),

    # Simpl_Array
    ("simpl_array", BENCH_DIR + "software/simpl_array/array_inc.mlir"),
    ("simpl_array", BENCH_DIR + "software/simpl_array/array_sum.mlir"),
    ("simpl_array", BENCH_DIR + "software/simpl_array/cube_in_place.mlir"),
    ("simpl_array", BENCH_DIR + "software/simpl_array/fourth_in_place.mlir"),
    ("simpl_array", BENCH_DIR + "software/simpl_array/sum_elts.mlir"),

    # Utdsp
    ("utdsp", BENCH_DIR + "software/utdsp/dct.mlir"),
    ("utdsp", BENCH_DIR + "software/utdsp/fir_small.mlir"),
    ("utdsp", BENCH_DIR + "software/utdsp/histogram.mlir"),
    ("utdsp", BENCH_DIR + "software/utdsp/lmsfir1.mlir"),
    ("utdsp", BENCH_DIR + "software/utdsp/lmsfir2.mlir"),
    ("utdsp", BENCH_DIR + "software/utdsp/mult_big.mlir"),
]

# fmt: on


def run_program(cmd, stdin=None):
    start = timer()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, input=stdin, stderr=subprocess.PIPE)
    end = timer()
    return p.stdout.decode("utf-8"), end - start, p.returncode


def main(argv):
    STATS_OUT_FILE = os.path.join(FLAGS.out_dir, "stats.csv")
    SYNTH_OUT_DIR = os.path.join(FLAGS.out_dir, "synth")

    if not os.path.exists(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)
    if not os.path.exists(SYNTH_OUT_DIR):
        os.makedirs(SYNTH_OUT_DIR)

    benchmarks = []
    if "polybench" in FLAGS.benchmarks:
        benchmarks += POLYBENCH
    if "software" in FLAGS.benchmarks:
        benchmarks += SOFTWARE

    if not benchmarks:
        print("No benchmarks selected")
        return

    for _ in tqdm.tqdm(range(FLAGS.n_iter)):
        for benchmark_suite, filename in benchmarks:
            cmd = ["timeout", str(FLAGS.timeout * 60)]
            cmd += [
                "python",
                SYNTHESIZER_PROGRAM,
                "--program",
                filename,
                "--target",
                FLAGS.target,
                "--synth_out",
                os.path.join(
                    SYNTH_OUT_DIR, os.path.basename(filename).replace(".mlir", ".out")
                ),
            ]
            if FLAGS.noguide:
                cmd += ["--noguide"]

            print(filename.replace(BENCH_DIR, ""), end="\t")

            out, runtime, exitcode = run_program(cmd)
            with open("/tmp/tensorize.log", "a") as f:
                f.write(out)

            # Print status code
            if exitcode == 0:
                print("\033[92m" + "OK" + "\033[0m", end="")
            else:
                print("\033[91m" + "Fail" + "\033[0m", end="")
            print("\t", end="")

            if exitcode != 0:
                print()
            else:
                # Print considered sketches
                statsStr = out.split("JSON: ")[1].split("\n")[0]
                stats = json.loads(statsStr)
                print(
                    "considered_sketches: %d, synth_time: %f"
                    % (stats["considered_sketches"], stats["synthesis_time"])
                )

                # Create result file if it does not exist
                stats_header = stats.keys()
                if not os.path.exists(STATS_OUT_FILE):
                    with open(STATS_OUT_FILE, "w") as f:
                        f.write(
                            "target,benchmark_suite,benchmark,runtime, "
                            + ",".join(stats_header)
                            + "\n"
                        )

                # Write to results file
                with open(STATS_OUT_FILE, "a") as f:
                    bench_name = filename.split("/")[-1].split(".")[0]
                    statsVals = ",".join([str(stats[key]) for key in stats_header])
                    f.write(
                        f"{FLAGS.target},{benchmark_suite},{bench_name},{runtime},{statsVals}\n"
                    )


if __name__ == "__main__":
    app.run(main)
