use ndarray::Array2;

pub trait Layer {
    // Initialize the layer with the input size (number of neurons in the previous layer).
    fn setup_layer(&mut self, num_in: usize);

    // Pass input forward through a series of operations.
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;

    // Pass output_grad backward through a series of operations.
    fn backward(&mut self, output_grad: Array2<f64>) -> Array2<f64>;

    // Collect parameter gradients from the layer's operations.
    fn params_grad(&mut self) -> Vec<Array2<f64>>;

    // Collect the parameters from the layer's operations.
    fn params(&mut self) -> Vec<Array2<f64>>;
}
