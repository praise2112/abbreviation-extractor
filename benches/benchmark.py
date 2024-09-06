import multiprocessing as mp
import random
import string
import time

import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from abbreviation_extractor import extract_abbreviation_definition_pairs, extract_abbreviation_definition_pairs_parallel
from abbreviations import schwartz_hearst
from scispacy.abbreviation import AbbreviationDetector

# Load spaCy model and add abbreviation detector
AbbreviationDetector
nlp = spacy.load("en_core_sci_sm")
# Disable all pipes except for the abbreviation detector
disabled_pipes = [pipe for pipe in nlp.pipe_names if pipe != "abbreviation_detector"]
nlp.disable_pipes(*disabled_pipes)
# Add the abbreviation detector if it's not already present
if "abbreviation_detector" not in nlp.pipe_names:
    nlp.add_pipe("abbreviation_detector")


def run_abbrv_schwartz_hearst(content):
    return schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=content)


def run_scispacy(content):
    doc = nlp(content)
    return {abrv.text: abrv._.long_form.text for abrv in doc._.abbreviations}


def run_abbreviation_extractor(content):
    extracted_pairs = extract_abbreviation_definition_pairs(content, False, False, True)
    return {pair.abbreviation: pair.definition for pair in extracted_pairs}


def run_abbreviation_extractor_parallel(texts):
    extracted_pairs = extract_abbreviation_definition_pairs_parallel(texts, False, False, True)
    return {pair.abbreviation: pair.definition for pair in extracted_pairs}


def warmup(libraries, warmup_text):
    for lib_name, lib_func in libraries.items():
        lib_func(warmup_text)


def benchmark_single_thread(func, text):
    start_time = time.time()
    func(text)
    end_time = time.time()
    return end_time - start_time  # Return throughput (texts/second)


def benchmark_multiprocessing(func, text):
    texts = [text] * BATCH_SIZE  # Create multiple copies of the text
    if func == run_abbreviation_extractor_parallel:
        start_time = time.time()
        func(texts)
        end_time = time.time()
    else:
        start_time = time.time()
        with mp.Pool(max_workers=mp.cpu_count()) as pool:
            pool.map(func, texts)
        end_time = time.time()
    return end_time - start_time


def generate_random_words(n):
    return ' '.join(''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))) for _ in range(n))


def generate_abbrev_pair():
    words = [generate_random_words(1).capitalize() for _ in range(random.randint(2, 4))]
    definition = ' '.join(words)
    abbrev = ''.join(word[0] for word in words)
    return f"{definition} ({abbrev})"


def generate_text(words_per_segment, abbrev_density=0.1):
    num_pairs = max(1, int(words_per_segment * abbrev_density / 10))  # Assuming each pair is about 10 words
    pairs = [generate_abbrev_pair() for _ in range(num_pairs)]
    text_segments = [generate_random_words(random.randint(5, 15)) for _ in range(num_pairs + 1)]
    full_text = ' '.join(text_segments[i] + ' ' + pairs[i] + ' ' for i in range(num_pairs)) + text_segments[-1]
    words = full_text.split()
    return ' '.join(words[:words_per_segment])


def generate_text_segments(max_words, step_size, abbrev_density=0.1):
    segments = []
    for word_count in range(step_size, max_words + 1, step_size):
        segments.append(generate_text(word_count, abbrev_density))
    return segments


def run_benchmarks(texts, libraries, n_runs, is_multiprocessing=False):
    results = {lib: [] for lib in libraries}

    for text in texts:
        for lib_name, lib_func in libraries.items():
            throughputs = []
            for _ in range(n_runs):
                if is_multiprocessing:
                    if lib_name == 'abbreviation_extractor (this)':
                        throughput = benchmark_multiprocessing(run_abbreviation_extractor_parallel, text)
                    else:
                        throughput = benchmark_multiprocessing(lib_func, text)
                else:
                    throughput = benchmark_single_thread(lib_func, text)
                throughputs.append(throughput)
            avg_throughput = sum(throughputs) / n_runs
            results[lib_name].append(avg_throughput)

    return results


def plot_results(single_results, multi_results, word_counts):
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 13))  # Increased figure height

    for lib, execution_times in single_results.items():
        sns.lineplot(x=word_counts, y=execution_times, marker='o', label=lib, ax=ax1)

    ax1.set_title("Single-Threaded Performance", fontsize=16, fontweight='bold', pad=10)
    ax1.set_xlabel("Word Count", fontsize=12)
    ax1.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax1.legend(fontsize=10)

    for lib, execution_times in multi_results.items():
        sns.lineplot(x=word_counts, y=execution_times, marker='o', label=lib, ax=ax2)

    ax2.set_title(f"Multi-Processing Performance (Batch size: {BATCH_SIZE})", fontsize=16, fontweight='bold', pad=10)
    ax2.set_xlabel("Word Count", fontsize=12)
    ax2.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax2.legend(fontsize=10)

    main_title = "Abbreviation Extraction Benchmark"
    subtitle = f"Abbreviation Density: {ABBREVIATION_DENSITY:.1%} | Runs per data point: {N_RUNS}"
    fig.suptitle(main_title, fontsize=22, fontweight='bold', y=0.98)
    fig.text(0.5, 0.945, subtitle, fontsize=14, ha='center')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.3)  # Adjusted top margin and vertical space between subplots
    plt.savefig('abbreviation_extraction_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()


def main(max_words=10000, step_size=1000, n_runs=5, abbrev_density=0.1):
    libraries = {
        'scispacy': run_scispacy,
        'abbreviation-extraction': run_abbrv_schwartz_hearst,
        'abbreviation_extractor (this)': run_abbreviation_extractor
    }

    texts = generate_text_segments(max_words, step_size, abbrev_density)

    with open('t.txt', 'w') as f:
        f.write(texts[-1])
    warmup(libraries, texts[0])
    word_counts = list(range(step_size, max_words + 1, step_size))

    single_results = run_benchmarks(texts, libraries, n_runs)
    multi_results = run_benchmarks(texts, libraries, n_runs)

    plot_results(single_results, multi_results, word_counts)


BATCH_SIZE = 500
ABBREVIATION_DENSITY = 0.05
N_RUNS = 5

if __name__ == "__main__":
    main(max_words=10000, step_size=500, n_runs=N_RUNS, abbrev_density=ABBREVIATION_DENSITY)
    # main(max_length=10000, step_size=1000, n_runs=5)
