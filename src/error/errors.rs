use std::fmt::Display;

#[derive(Debug, Clone)]
pub struct ShapeError {
    pub(crate) expected: Vec<usize>,
    pub(crate) found: Vec<usize>,
}

impl ShapeError {
    fn new(expected: Vec<usize>, found: Vec<usize>) -> Self {
        ShapeError { expected, found }
    }
}

impl Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Expected shape {:?}, but found shape {:?}",
            &self.expected, &self.found
        )
    }
}
