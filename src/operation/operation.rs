use crate::error::errors::ShapeError;
use crate::utils::utils::assert_same_shape_v2;
use ndarray::Array2;

pub struct Operation {
    pub(crate) input: Option<Array2<f64>>,
    pub(crate) output: Option<Array2<f64>>,
    pub(crate) input_grad: Option<Array2<f64>>,
}

impl Operation {
    pub fn new() -> Self {
        Self {
            input: None,
            output: None,
            input_grad: None,
        }
    }

    pub fn forward(&mut self, input: Array2<f64>) -> &Array2<f64> {
        self.input = Some(input);
        self.output = Some(self.output());

        self.output.as_ref().unwrap()
    }

    pub fn backward(&mut self, output_grad: &Array2<f64>) -> Result<&Array2<f64>, ShapeError> {
        assert_same_shape_v2(self.output.as_ref().unwrap(), output_grad)?;
        self.input_grad = Some(self.input_grad(output_grad));

        assert_same_shape_v2(
            self.input.as_ref().unwrap(),
            self.input_grad.as_ref().unwrap(),
        )?;

        Ok(self.input_grad.as_ref().unwrap())
    }

    fn output(&self) -> Array2<f64> {
        todo!("Implement this method")
    }

    pub(crate) fn input_grad(&self, output_grad: &Array2<f64>) -> Array2<f64> {
        todo!("Implement this method")
    }
}
