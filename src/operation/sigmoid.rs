use crate::operation::operation::Operation;
use ndarray::Array2;
use std::f64::consts::E;

pub struct Sigmod {
    pub operation: Operation,
}

impl Sigmod {
    pub fn new() -> Self {
        Self {
            operation: Operation::new(),
        }
    }

    pub fn output(&self) -> Array2<f64> {
        let input_ref = self.operation.input.as_ref().expect("Input not set.");
        let exponentiated = (-input_ref).mapv(|x| E.powf(x));
        1.0 / (1.0 + exponentiated)
    }
    pub fn input_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        let sigmoid_backward = self.operation.output.as_ref().unwrap()
            * (1.0 - self.operation.output.as_ref().unwrap());
        sigmoid_backward * output_grad
    }
}
