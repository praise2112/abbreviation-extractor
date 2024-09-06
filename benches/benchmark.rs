use abbreviation_extractor::{
    best_candidates, extract_abbreviation_definition_pairs, get_definition, select_definition,
    AbbreviationOptions, Candidate,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_best_candidates(c: &mut Criterion) {
    let sentence = "The World Health Organization (WHO) is a specialized agency of the United Nations (UN) responsible for international public health.";
    c.bench_function("best_candidates", |b| {
        b.iter(|| best_candidates(black_box(sentence)))
    });
}

fn bench_get_definition(c: &mut Criterion) {
    let sentence = "The World Health Organization (WHO) is a specialized agency.";
    let candidate = Candidate::new("WHO".to_string(), 31, 34);
    c.bench_function("get_definition", |b| {
        b.iter(|| get_definition(black_box(&candidate), black_box(sentence)))
    });
}

fn bench_select_definition(c: &mut Criterion) {
    let definition = Candidate::new("World Health and Organization".to_string(), 4, 31);
    let abbrev = "WHO";
    c.bench_function("select_definition", |b| {
        b.iter(|| select_definition(black_box(&definition), black_box(abbrev)))
    });
}

fn bench_extract_abbreviation_definition_pairs(c: &mut Criterion) {
    let text = "The World Health Organization (WHO) is a specialized agency of the United Nations (UN). WHO and UN work together on global health issues.";
    c.bench_function("extract_abbreviation_definition_pairs", |b| {
        b.iter(|| {
            extract_abbreviation_definition_pairs(black_box(&text), AbbreviationOptions::default())
        })
    });
}

fn bench_extract_abbreviation_definition_pairs_with_tokenizer(c: &mut Criterion) {
    let text = "The World Health Organization (WHO) is a specialized agency of the United Nations (UN). WHO and UN work together on global health issues.";
    let options = AbbreviationOptions::new(false, false, true);
    c.bench_function(
        "extract_abbreviation_definition_pairs_with_tokenizer",
        |b| b.iter(|| extract_abbreviation_definition_pairs(black_box(text), options)),
    );
}

fn bench_extract_abbreviation_definition_pairs_long_multiline_text(c: &mut Criterion) {
    let text = "The World Health Organization (WHO) is a specialized agency of the United Nations (UN). \n\
     WHO and UN work together on global health issues. The World Health Organization (WHO) is a specialized agency of the United Nations (UN). WHO and UN work together on global health issues.\n\
      The World Health Organization (WHO) is a specialized agency of the United Nations (UN). WHO and UN work together on global health issues. \n\
      The World Health Organization (WHO) is a specialized agency of the United Nations (UN).\n\
       WHO and UN work together on global health issues. The World Health Organization (WHO) is a specialized agency of the United Nations (UN). \n\
       WHO and UN work together on global health issues.";
    let options = AbbreviationOptions::new(false, false, true);
    c.bench_function(
        "extract_abbreviation_definition_pairs_long_multiline_text",
        |b| b.iter(|| extract_abbreviation_definition_pairs(black_box(text), options)),
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(100).measurement_time(std::time::Duration::from_secs(20));
    // targets = bench_select_definition
    // targets =  bench_extract_abbreviation_definition_pairs
    // targets =  bench_extract_abbreviation_definition_pairs, bench_extract_abbreviation_definition_pairs_2, bench_extract_abbreviation_definition_pairs_with_tokenizer
    targets = bench_best_candidates, bench_get_definition, bench_select_definition, bench_extract_abbreviation_definition_pairs, bench_extract_abbreviation_definition_pairs_with_tokenizer, bench_extract_abbreviation_definition_pairs_long_multiline_text
}
criterion_main!(benches);
