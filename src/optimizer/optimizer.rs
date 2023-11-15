pub trait Optimizer {
    // Every optimizer must implement the "step" function.
    fn step(&self);
}
