use ndarray::Array2;

pub trait Operation {
    fn forward(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String>;
    fn backward(&mut self, output_grad: Array2<f64>) -> Result<Array2<f64>, String>;
    fn param_grads(&self) -> Vec<Array2<f64>>;
}
