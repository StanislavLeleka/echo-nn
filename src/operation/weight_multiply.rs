use crate::operation::operation::Operation;
use ndarray::Array2;

pub struct WeightMultiply {
    pub weights: Array2<f64>,
    pub wight_grads: Option<Array2<f64>>,
    pub input: Option<Array2<f64>>,
}

impl WeightMultiply {
    pub fn new(weights: Array2<f64>) -> Self {
        Self {
            weights,
            input: None,
            wight_grads: None,
        }
    }
}

impl Operation for WeightMultiply {
    fn forward(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String> {
        self.input = Some(input);
        let input_ref = self.input.as_ref().ok_or("Input not set.")?;
        Ok(input_ref.dot(&self.weights))
    }

    fn backward(&mut self, output_grad: Array2<f64>) -> Result<Array2<f64>, String> {
        let input_ref = self
            .input
            .as_ref()
            .ok_or("No input cache available for backward pass.")?;
        let input_grad = output_grad.dot(&self.weights.t());
        let weights_grad = input_ref.t().dot(&output_grad);

        // Save the weights gradient for parameter updates
        self.wight_grads = Some(weights_grad);

        Ok(input_grad)
    }
}
