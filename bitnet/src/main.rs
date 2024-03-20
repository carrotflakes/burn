use bitnet::run;

fn main() {
    run("/tmp/text-generation", "[START]how".to_owned(), 64);
    // run("./text-generation", "The US".to_owned(), 100);
}
