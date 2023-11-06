use ndarray::{Array2, Axis};

pub enum OperationType {
    WeightMultiply { weights: Array2<f64> },
    BiasAdd { biases: Array2<f64> },
    Sigmoid,
}

pub struct Operation {
    pub(crate) input: Option<Array2<f64>>,
    pub(crate) output: Option<Array2<f64>>,
    pub(crate) operation_type: OperationType,
}

impl Operation {
    pub fn weight_multiply(weights: Array2<f64>) -> Self {
        Self {
            input: Some(Array2::zeros(weights.dim())),
            output: Some(Array2::zeros(weights.dim())),
            operation_type: OperationType::WeightMultiply { weights },
        }
    }

    pub fn bias_add(biases: Array2<f64>) -> Self {
        Self {
            input: Some(Array2::zeros(biases.dim())),
            output: Some(Array2::zeros(biases.dim())),
            operation_type: OperationType::BiasAdd { biases },
        }
    }

    pub fn sigmoid() -> Self {
        Self {
            input: Some(Array2::zeros((1, 1))),
            output: Some(Array2::zeros((1, 1))),
            operation_type: OperationType::Sigmoid,
        }
    }

    /// Performs the forward pass for the operation.
    pub fn forward(&mut self, input: Array2<f64>) -> Result<&Option<Array2<f64>>, String> {
        self.input = Some(input);

        match self.operation_type {
            OperationType::WeightMultiply { ref weights } => {
                let input_ref = self.input.as_ref().ok_or("Input not set.")?;
                self.output = Some(input_ref.dot(weights));
            }
            OperationType::BiasAdd { ref biases } => {
                let input_ref = self.input.as_ref().ok_or("Input not set.")?;
                if input_ref.shape()[0] != biases.shape()[0] {
                    return Err("Shapes do not match for Bias Addition.".to_string());
                }
                self.output = Some(input_ref + biases);
            }
            OperationType::Sigmoid => {
                let input_ref = self.input.as_ref().ok_or("Input not set.")?;
                self.output = Some(input_ref.mapv(|x| 1.0 / (1.0 + (-x).exp())));
            }
        }

        Ok(&self.output)
    }

    /// Performs the backward pass of the operation.
    pub fn backward(&mut self, output_grad: Array2<f64>) -> Result<Array2<f64>, String> {
        let output_ref = self
            .output
            .as_ref()
            .ok_or("No output available for backward pass.")?;

        if output_ref.shape() != output_grad.shape() {
            return Err("Output and output gradient shapes do not match.".to_string());
        }

        match &self.operation_type {
            OperationType::WeightMultiply { weights } => {
                // Compute the gradient with respect to input
                let input_grad = output_grad.dot(&weights.t());
                Ok(input_grad)
            }
            OperationType::BiasAdd { biases } => {
                if output_grad.shape()[1] != biases.shape()[1] {
                    return Err("Bias gradient shape mismatch.".to_string());
                }
                // The gradient for bias is the sum over axis 0
                let sum_grad = output_grad.sum_axis(Axis(0));
                Ok(Array2::from_shape_vec((1, sum_grad.len()), sum_grad.to_vec()).unwrap())
            }
            OperationType::Sigmoid => {
                // Compute the gradient for sigmoid function
                let sigmoid_grad = output_ref * (&Array2::ones(output_ref.raw_dim()) - output_ref);
                Ok(sigmoid_grad * &output_grad)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::operation::operation::{Operation, OperationType};
    use ndarray::array;

    #[test]
    fn test_backward_weights() {
        // Create an instance of Operation for weights with a dummy value
        let mut operation = Operation {
            input: Some(array![[1.0, 2.0], [3.0, 4.0]]),
            output: Some(array![[2.0, 3.0], [4.0, 5.0]]),
            operation_type: OperationType::WeightMultiply {
                weights: array![[1.0, 0.0], [0.0, 1.0]],
            },
        };

        // Calculate the backward pass
        let output_grad = array![[1.0, 0.0], [0.0, 1.0]];
        let input_grad = operation.backward(output_grad).unwrap();

        // Check the correctness of the backward pass
        assert_eq!(input_grad, array![[1.0, 0.0], [0.0, 1.0]]);
    }

    #[test]
    fn test_backward_bias() {
        // Create an instance of Operation for bias with a dummy value
        let mut operation = Operation {
            input: Some(array![[1.0], [1.0]]),
            output: Some(array![[3.0], [3.0]]),
            operation_type: OperationType::BiasAdd {
                biases: array![[2.0]],
            },
        };

        // Calculate the backward pass
        let output_grad = array![[1.0], [1.0]];
        let input_grad = operation.backward(output_grad).unwrap();

        // Check that the sum of the gradients is correct for bias elements
        assert_eq!(input_grad, array![[2.0]]);
    }

    #[test]
    fn test_backward_sigmoid() {
        // Create an instance of Operation for sigmoid with a dummy value
        let mut operation = Operation {
            input: Some(array![[0.0]]),
            output: Some(array![[0.5]]),
            operation_type: OperationType::Sigmoid,
        };

        // Calculate the backward pass
        let output_grad = array![[1.0]];
        let input_grad = operation.backward(output_grad).unwrap();

        // The gradient for sigmoid when the input is 0 should be 0.25
        assert_eq!(input_grad, array![[0.25]]);
    }
}
