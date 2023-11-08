use crate::loss::loss::Loss;
use ndarray::Array2;
use ndarray_rand::rand_distr::num_traits::real::Real;

pub struct MeanSquaredError {
    prediction: Array2<f64>,
    target: Array2<f64>,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {
            prediction: Array2::zeros((0, 0)), // Placeholder, actual size set in forward
            target: Array2::zeros((0, 0)),     // Placeholder, actual size set in forward
        }
    }
}

impl Loss for MeanSquaredError {
    fn forward(&mut self, prediction: &Array2<f64>, target: &Array2<f64>) -> f64 {
        // Ensure the predictions and targets have the same shape.
        assert_eq!(prediction.shape(), target.shape());

        self.prediction = prediction.clone();
        self.target = target.clone();
        self.output()
    }

    fn backward(&mut self) -> Array2<f64> {
        // Computes gradient of the loss value with respect to the input to the loss function.
        self.input_grad()
    }

    fn output(&self) -> f64 {
        // Calculate the mean squared error.
        let diff = &self.prediction - &self.target;
        let error = diff.mapv(|x| x.powi(2)).sum_axis(ndarray::Axis(0));
        error.sum() / self.prediction.nrows() as f64
    }

    fn input_grad(&self) -> Array2<f64> {
        // Calculate the gradient for the mean squared error.
        let diff = &self.prediction - &self.target;
        2.0 * diff / self.prediction.nrows() as f64
    }
}
