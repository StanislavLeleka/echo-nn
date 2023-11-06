use crate::operation::operation::Operation;
use ndarray::Array2;

pub struct Sigmoid {
    pub output: Option<Array2<f64>>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Operation for Sigmoid {
    fn forward(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String> {
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output = Some(output.clone());
        Ok(output)
    }

    fn backward(&mut self, output_grad: Array2<f64>) -> Result<Array2<f64>, String> {
        let output = self
            .output
            .as_ref()
            .ok_or("No output cache available for backward pass.")?;
        // Sigmoid derivative is output * (1.0 - output)
        let sigmoid_grad = output * (&Array2::ones(output.raw_dim()) - output);
        // Element-wise multiplication of sigmoid_grad and output_grad
        let input_grad = sigmoid_grad * &output_grad;
        Ok(input_grad)
    }
}
