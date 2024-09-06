use std::borrow::Cow;

#[derive(Debug, Clone, PartialEq)]
pub struct Candidate<'a> {
    text: Cow<'a, str>,
    start: usize,
    stop: usize,
}

impl<'a> Candidate<'a> {
    pub fn new(text: impl Into<Cow<'a, str>>, start: usize, stop: usize) -> Self {
        Candidate {
            text: text.into(),
            start,
            stop,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn stop(&self) -> usize {
        self.stop
    }

    pub fn set_position(&mut self, start: usize, stop: usize) {
        self.start = start;
        self.stop = stop;
    }
}
