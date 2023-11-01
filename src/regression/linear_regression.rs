use ndarray::{s, Array2, Axis};
use ndarray_rand::{rand_distr::Normal, RandomExt};
use plotters::{
    prelude::{BitMapBackend, Circle, EmptyElement, IntoDrawingArea, PathElement},
    series::{LineSeries, PointSeries},
    style::{Color, IntoFont, ShapeStyle, BLUE, RED, WHITE},
};
use rand::{distributions::Standard, prelude::Distribution, seq::SliceRandom};
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
        let loss = (y_batch - &p).mapv(|x| x.powi(2)).sum() / y_batch.dim().0 as f64;

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

        let dl_dp = -2.0 * (&forward_pass_variables["y_batch"] - &forward_pass_variables["p"]);

        let dp_dn = Array2::from_elem(dl_dp.dim(), 1.0);
        let dl_dn = dl_dp.clone() * &dp_dn;

        let dl_dw = forward_pass_variables["x_batch"].t().dot(&dl_dn);

        // Compute gradient with respect to b.
        // Sum over all examples because b is a scalar.
        let dl_db_1d = dl_dp.sum_axis(Axis(0));
        let dl_db = dl_db_1d.insert_axis(Axis(1)); // Convert to 2D array

        let mut gradients = HashMap::new();
        gradients.insert("w".to_string(), dl_dw);
        gradients.insert("b".to_string(), dl_db);

        gradients
    }

    fn init_weights(&self, n_in: usize) -> HashMap<String, Array2<f64>> {
        // Initialize weights for the step-by-step linear regression model.

        let w = Array2::from_elem((n_in, 1), 0.0);
        let b = Array2::from_elem((1, 1), 0.0);

        let mut weights = HashMap::new();
        weights.insert("w".to_string(), w);
        weights.insert("b".to_string(), b);

        weights
    }

    fn permute_data(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, Array2<f64>) {
        assert_eq!(x.dim().0, y.dim().0);

        // Create a random permutation of indices.
        let mut indices: Vec<usize> = (0..x.dim().0).collect();
        indices.shuffle(&mut rand::thread_rng());

        // Permute the rows of X and y using the permutation.
        let x_permuted = x.select(Axis(0), &indices);
        let y_permuted = y.select(Axis(0), &indices);

        (x_permuted, y_permuted)
    }

    fn generate_batch(
        &self,
        x: Array2<f64>,
        y: Array2<f64>,
        start: usize,
        batch_size: usize,
    ) -> (Array2<f64>, Array2<f64>) {
        // Generate a batch of data.

        assert_eq!(x.dim().0, y.dim().0);

        let end = min(start + batch_size, x.dim().0);

        let x_batch = x.slice(s![start..end, ..]).to_owned();
        let y_batch = y.slice(s![start..end, ..]).to_owned();

        (x_batch, y_batch)
    }

    fn train(
        &self,
        x: Array2<f64>,
        y: Array2<f64>,
        epochs: u32,
        learning_rate: f64,
        batch_size: usize,
    ) -> (Vec<f64>, HashMap<String, Array2<f64>>) {
        // Train the model.

        // Initialize weights.
        let mut weights = self.init_weights(x.shape()[1] as usize);

        // Permute data.
        let (mut x_permuted, mut y_permuted) = self.permute_data(&x, &y);

        let mut losses = Vec::new();

        // Iterate over epochs.
        let mut start = 0;
        for _ in 0..epochs {
            // Generate a batch.
            let (x_batch, y_batch) =
                self.generate_batch(x_permuted.clone(), y_permuted.clone(), start, batch_size);

            start += batch_size;

            if start >= x_permuted.dim().0 {
                // Permute data for the next epoch.
                let permuted = self.permute_data(&x, &y);
                x_permuted = permuted.0;
                y_permuted = permuted.1;
                start = 0;
            }

            // Compute the forward pass.
            let (loss, forward_pass_variables) =
                self.forward_linear_regression(&x_batch, &y_batch, weights.clone());

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

    fn predict(X: &Array2<f64>, weights: HashMap<String, Array2<f64>>) -> Array2<f64> {
        // Predict using the step-by-step linear regression model.

        let n = X.dot(&weights["w"]);
        let p = n.clone() + &weights["b"];

        p
    }

    fn mae(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        // Compute the mean absolute error.

        assert_eq!(y_true.dim(), y_pred.dim());

        let mae = (y_true - y_pred).mapv(|x| x.abs()).sum() / y_true.dim().0 as f64;

        mae
    }

    fn rmse(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> f64 {
        // Compute the root mean squared error.

        assert_eq!(y_true.dim(), y_pred.dim());

        let rmse = ((y_true - y_pred).mapv(|x| x.powi(2)).sum() / y_true.dim().0 as f64).sqrt();

        rmse
    }

    fn plot_losses(losses: &Vec<f64>) {
        let root =
            BitMapBackend::new("src/data/linear_regression.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let max_loss = *losses
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        let min_loss = *losses
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);

        let mut chart = plotters::chart::ChartBuilder::on(&root)
            .caption("Training Loss Over Epochs", ("sans-serif", 20).into_font())
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(0..losses.len(), min_loss..max_loss)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(
                losses.iter().enumerate().map(|(x, y)| (x, *y)),
                &RED,
            ))
            .unwrap()
            .label("Loss")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels().draw().unwrap();
    }

    fn plot_predictions(x: &Array2<f64>, y_true: &Array2<f64>, y_pred: &Array2<f64>) {
        let root = BitMapBackend::new("src/data/linear_regression_predictions.png", (800, 600))
            .into_drawing_area();
        root.fill(&WHITE).unwrap();

        let x_values: Vec<f64> = x.column(0).to_vec();
        let y_true_values: Vec<f64> = y_true.column(0).to_vec();
        let y_pred_values: Vec<f64> = y_pred.column(0).to_vec();

        println!("x_values: {:?}", x_values);
        println!("y_true_values: {:?}", y_true_values);
        println!("y_pred_values: {:?}", y_pred_values);

        let x_min = x
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            - 5.0;
        let x_max = x
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            + 5.0;
        let y_min = y_true
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            - 5.0;
        let y_max = y_true
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0)
            + 5.0;

        let mut chart = plotters::chart::ChartBuilder::on(&root)
            .caption("True vs Predicted", ("sans-serif", 40))
            .margin(5)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(
                x_values
                    .iter()
                    .zip(&y_true_values)
                    .map(|(&x, &y)| Circle::new((x, y), 5, BLUE.filled())),
            )
            .unwrap();

        chart
            .draw_series(LineSeries::new(
                x_values.iter().zip(&y_pred_values).map(|(&x, &y)| (x, y)),
                &RED,
            ))
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use std::collections::HashMap;

    use super::LinearRegression;

    #[test]
    fn test_linear_regression() {
        let (x, y) = read_salary_dataset();

        let linear_regression = LinearRegression::new();

        let (losses, weights) = linear_regression.train(x.clone(), y.clone(), 50, 0.0001, 32);

        println!("weights: {:?}", weights);

        let y_pred = LinearRegression::predict(&x, weights);

        let mae = LinearRegression::mae(&y, &y_pred);
        let rmse = LinearRegression::rmse(&y, &y_pred);

        println!("x: {:?}", x);
        println!("y: {:?}", y);
        println!("y_pred: {:?}", y_pred);

        println!("MAE: {}", mae);
        println!("RMSE: {}", rmse);

        LinearRegression::plot_losses(&losses);
        LinearRegression::plot_predictions(&x, &y, &y_pred);
    }

    #[test]
    fn test_linear_regression2() {
        let x = Array2::from_shape_vec((50, 1), (0..50).map(|x| x as f64).collect()).unwrap();
        let y = &x * 2.0 + 5.0; // Simple linear relation
        let model = LinearRegression::new();
        let (losses, weights) = model.train(x.clone(), y.clone(), 200, 0.0001, 10);
        let y_pred = LinearRegression::predict(&x, weights);
        let mae = LinearRegression::mae(&y, &y_pred);
        let rmse = LinearRegression::rmse(&y, &y_pred);

        println!("mae: {}", mae);
        println!("rmse: {}", rmse);

        LinearRegression::plot_predictions(&x, &y, &y_pred);
        LinearRegression::plot_losses(&losses);
    }

    fn read_salary_dataset() -> (Array2<f64>, Array2<f64>) {
        let mut rdr = csv::Reader::from_path("src/data/salary_dataset.csv").unwrap();

        let mut data = Vec::new();
        let mut target = Vec::new();

        for result in rdr.records() {
            let record = result.unwrap();
            let x = record[1].parse::<f64>().unwrap();
            let y = record[2].parse::<f64>().unwrap();

            data.push(x);
            target.push(y);
        }

        let data_array = Array2::from_shape_vec((data.len(), 1), data).unwrap();
        let target_array = Array2::from_shape_vec((target.len(), 1), target).unwrap();

        (data_array, target_array)
    }

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
        assert!(gradients["w"].mapv(|x| x.abs()).sum() < tolerance);
        assert!(gradients["b"].mapv(|x| x.abs()).sum() < tolerance);
    }
}
