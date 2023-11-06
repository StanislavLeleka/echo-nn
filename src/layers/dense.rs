use crate::layers::layer::Layer;
use crate::operations::bias_add::BiasAdd;
use crate::operations::operation::Operation;
use crate::operations::weight_multiply::WeightMultiply;
use ndarray::Array2;
use ndarray_rand::rand_distr::Normal;
use rand::distributions::Distribution;

pub struct Dense {
    neurons: usize,
    first: bool,
    params: Vec<Array2<f64>>,
    param_grads: Vec<Array2<f64>>,
    operations: Vec<Box<dyn Operation>>,
    activation: Box<dyn Operation>,
}

impl Dense {
    pub fn new(neurons: usize, activation: Box<dyn Operation>) -> Self {
        Self {
            neurons,
            first: true,
            params: Vec::new(),
            param_grads: Vec::new(),
            operations: Vec::new(),
            activation,
        }
    }
}

impl Layer for Dense {
    fn setup_layer(&mut self, num_in: usize) {
        if self.first {
            let mut rng = rand::thread_rng();
            let normal_dist = Normal::new(0.0, 1.0).unwrap(); // Standard normal distribution
            let weights =
                Array2::from_shape_fn((num_in, self.neurons), |_| normal_dist.sample(&mut rng));
            let biases = Array2::from_shape_fn((1, self.neurons), |_| normal_dist.sample(&mut rng));

            // Add the weight and bias operations to the operations vector.
            self.operations
                .push(Box::new(WeightMultiply::new(weights.clone())));
            self.operations.push(Box::new(BiasAdd::new(biases.clone())));

            // Store parameters for potential access (like for updating weights).
            self.params.push(weights);
            self.params.push(biases);

            self.first = false;
        }
    }

    fn forward(&mut self, mut input: Array2<f64>) -> Array2<f64> {
        if self.first {
            self.setup_layer(input.shape()[1]);
        }

        for operation in self.operations.iter_mut() {
            input = operation.forward(input).unwrap();
        }

        // Apply activation function
        self.activation.forward(input).unwrap()
    }

    fn backward(&mut self, mut output_grad: Array2<f64>) -> Array2<f64> {
        // Apply activation function gradient
        output_grad = self.activation.backward(output_grad).unwrap();

        for operation in self.operations.iter_mut().rev() {
            output_grad = operation.backward(output_grad).unwrap();
        }

        self.collect_param_grads();
        output_grad
    }

    fn params(&mut self) -> Vec<Array2<f64>> {
        self.params.clone()
    }

    fn params_grad(&mut self) -> Vec<Array2<f64>> {
        self.param_grads.clone()
    }
}

impl Dense {
    fn collect_param_grads(&mut self) {
        self.param_grads.clear();
        for operation in &self.operations {
            self.param_grads.extend(operation.param_grads());
        }
    }
}

#[cfg(test)]
mod test {
    use crate::layers::dense::Dense;
    use crate::layers::layer::Layer;
    use crate::operations::bias_add::BiasAdd;
    use crate::operations::sigmoid::Sigmoid;
    use crate::operations::weight_multiply::WeightMultiply;
    use ndarray::arr2;

    fn create_test_dense_layer() -> Dense {
        // Create dummy weights and biases.
        let weights = arr2(&[[0.1, 0.2], [0.3, 0.4]]);
        let biases = arr2(&[[0.01, 0.02]]);
        let weight_multiply_op = Box::new(WeightMultiply::new(weights));
        let bias_add_op = Box::new(BiasAdd::new(biases));
        let activation_op = Box::new(Sigmoid::new());

        // Create a DenseLayer with 2 neurons and a Sigmoid activation function.
        let mut dense_layer = Dense::new(2, activation_op);
        dense_layer.operations.push(weight_multiply_op);
        dense_layer.operations.push(bias_add_op);

        dense_layer
    }

    #[test]
    fn test_dense_layer_forward_pass() {
        let mut dense_layer = create_test_dense_layer();

        // Provide input to the layer.
        let input = arr2(&[[1.0, 1.0]]);
        // Perform a forward pass.
        let output = dense_layer.forward(input);

        // Check the output dimensions.
        assert_eq!(output.shape(), &[1, 2]);

        // Check the values of the output (assuming we know what they should be).
        // Here we're just checking that the output is not the same as the input.
        assert_ne!(output, arr2(&[[1.0, 1.0]]));
    }

    #[test]
    fn test_dense_layer_backward_pass() {
        let mut layer = create_test_dense_layer();
        // Provide a dummy gradient from the next layer or loss function.
        let gradient = arr2(&[[0.1, -0.1]]);
        // Perform a forward pass first (needed to populate caches).
        let _ = layer.forward(arr2(&[[1.0, 1.0]]));
        // Perform a backward pass.
        let input_grad = layer.backward(gradient);

        // Check the input gradient dimensions.
        assert_eq!(input_grad.shape(), &[1, 2]);

        // Verify that gradients are propagated.
        assert_ne!(input_grad, arr2(&[[0.0, 0.0]]));
    }

    #[test]
    fn test_dense_layer_param_grads() {
        let mut layer = create_test_dense_layer();
        // Perform a forward pass first.
        let _ = layer.forward(arr2(&[[1.0, 1.0]]));
        // Perform a backward pass to compute gradients.
        let _ = layer.backward(arr2(&[[0.1, -0.1]]));
        // Get parameter gradients.
        let param_grads = layer.params_grad();

        // Check that we have gradients for both weights and biases.
        // assert_eq!(param_grads.len(), 2);
        // Check the dimensions of the weight and bias gradients.
        assert_eq!(param_grads[0].shape(), &[2, 2]);
        assert_eq!(param_grads[1].shape(), &[1, 2]);
    }
}
