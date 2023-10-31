use ndarray::{s, Array2, Axis};
use ndarray_rand::rand_distr::Normal;
use rand::{prelude::Distribution, seq::SliceRandom};
use std::{cmp::min, collections::HashMap};

pub struct LinearRegression {}

impl LinearRegression {
    pub fn new() -> LinearRegression {
        LinearRegression {}
    }

    fn forward_linear_regression(
        &self,
        x_batch: &Array2<f64>,
        y_batch: &Array2<f64>,
        weights: HashMap<String, Array2<f64>>,
    ) -> (f64, HashMap<String, Array2<f64>>) {
        // Forward pass for the step-by-step linear regression.

        // Assert that the batch size is the same for x and y.
        assert_eq!(x_batch.shape()[0], y_batch.shape()[0]);

        // Assert that the number of features is the same for x and w.
        assert_eq!(x_batch.shape()[1], weights["w"].shape()[0]);

        // Compute the forward pass.
        let n = x_batch.dot(&weights["w"]);
        let p = n.clone() + &weights["b"];

        // Compute the loss.
        let loss = (y_batch - &p).mapv(|x| x.powi(2)).sum() / (2.0 * x_batch.shape()[0] as f64);

        // Save the forward pass variables.
        let mut forward_pass_variables = HashMap::new();
        forward_pass_variables.insert("n".to_string(), n);
        forward_pass_variables.insert("p".to_string(), p);
        forward_pass_variables.insert("x_batch".to_string(), x_batch.clone());
        forward_pass_variables.insert("y_batch".to_string(), y_batch.clone());

        return (loss, forward_pass_variables);
    }

    fn loss_gradient(
        &self,
        forward_pass_variables: HashMap<String, Array2<f64>>,
        weights: HashMap<String, Array2<f64>>,
    ) -> HashMap<String, Array2<f64>> {
        // Compute dLdW and dLdB for the step-by-step linear regression model.

        let dLdP = -2.0 * (&forward_pass_variables["y_batch"] - &forward_pass_variables["p"]);

        // Compute gradient with respect to W.
        let dLdW = forward_pass_variables["x_batch"].t().dot(&dLdP);

        // Compute gradient with respect to b.
        // Sum over all examples because b is a scalar.
        let dLdB_sum = dLdP.sum();
        let dLdB = Array2::from_elem((1, 1), dLdB_sum);

        let mut gradients = HashMap::new();
        gradients.insert("dLdW".to_string(), dLdW);
        gradients.insert("dLdB".to_string(), dLdB);

        return gradients;
    }

    fn init_weights(&self, n_in: u32) -> HashMap<String, Array2<f64>> {
        // Initialize weights for the step-by-step linear regression model.

        let normal = Normal::new(0.0, 1.0).unwrap();

        let w = Array2::from_shape_fn((n_in as usize, 1), |_| {
            normal.sample(&mut rand::thread_rng())
        });
        let b = Array2::from_shape_fn((1, 1), |_| normal.sample(&mut rand::thread_rng()));

        let mut weights = HashMap::with_capacity(2);
        weights.insert("W".to_string(), w);
        weights.insert("B".to_string(), b);

        return weights;
    }

    fn permute_data(&self, X: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        assert_eq!(X.dim().0, y.dim().0);

        // Create a random permutation of indices.
        let mut indices: Vec<usize> = (0..X.dim().0).collect();
        indices.shuffle(&mut rand::thread_rng());

        // Permute the rows of X and y using the permutation.
        let X_permuted = X.select(Axis(0), &indices);
        let y_permuted = y.select(Axis(0), &indices);

        (X_permuted, y_permuted)
    }

    fn generate_batch(
        &self,
        X: Array2<f64>,
        y: Array2<f64>,
        start: usize,
        batch_size: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        // Generate a batch of data.

        assert_eq!(X.dim().0, y.dim().0);

        let end = min(start + batch_size, X.dim().0);

        let X_batch = X.slice(s![start..end, ..]).to_owned();
        let y_batch = y.slice(s![start..end, ..]).to_owned();

        (X_batch, y_batch)
    }

    fn train(
        &self,
        X: Array2<f64>,
        y: Array2<f64>,
        epochs: u32,
        learning_rate: f64,
        batch_size: usize,
    ) -> (Vec<f64>, HashMap<String, Array2<f64>>) {
        // Train the model.

        // Initialize weights.
        let mut weights = self.init_weights(X.shape()[1] as u32);

        // Permute data.
        let (mut X_permuted, mut y_permuted) = self.permute_data(&X, &y);

        let mut losses = Vec::new();

        // Iterate over epochs.
        let mut start = 0;
        for _ in 0..epochs {
            // Generate a batch.
            let (X_batch, y_batch) =
                self.generate_batch(X_permuted.clone(), y_permuted.clone(), start, batch_size);

            start += batch_size;

            if start >= X_permuted.dim().0 {
                // Permute data for the next epoch.
                let permuted = self.permute_data(&X, &y);
                X_permuted = permuted.0;
                y_permuted = permuted.1;
                start = 0;
            }

            // Compute the forward pass.
            let (loss, forward_pass_variables) =
                self.forward_linear_regression(&X_batch, &y_batch, weights.clone());

            losses.push(loss);

            // Compute the gradients.
            let gradients = self.loss_gradient(forward_pass_variables, weights.clone());

            // Update the weights.
            for (key, value) in &gradients {
                let update = &weights[key] - learning_rate * value;
                weights.insert(key.clone(), update);
            }
        }

        (losses, weights)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use std::collections::HashMap;

    use super::LinearRegression;

    #[test]
    fn test_forward_linear_regression() {
        let x_batch = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_batch = array![[5.0], [11.0], [17.0]]; // For simplicity, let's assume y = x1 + 2*x2
        let mut weights = HashMap::new();

        weights.insert("w".to_string(), array![[1.0], [2.0]]);
        weights.insert("b".to_string(), array![[0.0]]);

        let (loss, _) =
            LinearRegression::new().forward_linear_regression(&x_batch, &y_batch, weights.clone());

        let tolerance = 1e-5;
        assert!((loss - 0.0).abs() < tolerance); // With the given weights and inputs, loss should be 0.
    }

    #[test]
    fn test_loss_gradient() {
        let x_batch = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_batch = array![[5.0], [11.0], [17.0]]; // For simplicity, let's assume y = x1 + 2*x2
        let mut weights = HashMap::new();

        weights.insert("w".to_string(), array![[1.0], [2.0]]);
        weights.insert("b".to_string(), array![[0.0]]);

        let linear_regression = LinearRegression::new();

        let (_, forward_pass_variables) =
            linear_regression.forward_linear_regression(&x_batch, &y_batch, weights.clone());

        let gradients = linear_regression.loss_gradient(forward_pass_variables, weights);

        let tolerance = 1e-5;
        assert!(gradients["dLdW"].mapv(|x| x.abs()).sum() < tolerance);
        assert!(gradients["dLdB"].mapv(|x| x.abs()).sum() < tolerance);
    }
}
