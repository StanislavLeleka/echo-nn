use ndarray::Array2;

pub trait Loss {
    // Computes the loss given the prediction and target.
    fn forward(&mut self, prediction: &Array2<f64>, target: &Array2<f64>) -> f64;

    // Computes the gradient of the loss with respect to the input.
    fn backward(&mut self) -> Array2<f64>;

    // The method that actually calculates the loss value.
    fn output(&self) -> f64;

    // The method that actually calculates the gradient of the loss.
    fn input_grad(&self) -> Array2<f64>;
}
