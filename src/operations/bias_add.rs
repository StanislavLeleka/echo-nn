use crate::operations::operation::Operation;
use ndarray::{Array2, Axis};

pub struct BiasAdd {
    pub biases: Array2<f64>,
    pub biases_grad: Option<Array2<f64>>,
}

impl BiasAdd {
    pub fn new(biases: Array2<f64>) -> Self {
        Self {
            biases,
            biases_grad: None,
        }
    }
}

impl Operation for BiasAdd {
    fn forward(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String> {
        if input.shape()[0] != self.biases.shape()[0] {
            return Err("Shapes do not match for Bias Addition.".to_string());
        }
        Ok(input + &self.biases)
    }

    fn backward(&mut self, output_grad: Array2<f64>) -> Result<Array2<f64>, String> {
        // The bias gradient is the sum of the gradients over all observations
        let biases_grad = output_grad.sum_axis(Axis(0)).insert_axis(Axis(0));
        // Save the biases gradient for parameter updates
        self.biases_grad = Some(biases_grad);
        Ok(output_grad) // The gradient doesn't change with respect to the biases
    }

    fn param_grads(&self) -> Vec<Array2<f64>> {
        match &self.biases_grad {
            Some(grads) => vec![grads.clone()],
            None => vec![],
        }
    }
}
