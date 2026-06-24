use crate::{
    gl::{
        GetError, INVALID_ENUM, INVALID_FRAMEBUFFER_OPERATION, INVALID_OPERATION, INVALID_VALUE,
        NO_ERROR, OUT_OF_MEMORY,
    },
    GLenum,
};

/// Error Object for OpenGL.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Error(GLenum);

impl Error {
    /// Create an new error with glGetError.
    pub fn new() -> Self {
        Self(unsafe { GetError() })
    }

    /// Return true if error raised.
    pub fn is_error(self) -> bool {
        self.0 != NO_ERROR
    }

    /// Return true if no error.
    pub fn is_okay(self) -> bool {
        self.0 == NO_ERROR
    }

    /// Return human readable text of the error code.
    pub fn to_str(self) -> &'static str {
        match self.0 {
            NO_ERROR => "No error",
            INVALID_ENUM => "An unacceptable value is specified for an enumerated argument",
            INVALID_VALUE => "A numeric argument is out of range",
            INVALID_OPERATION => "The specified operation is not allowed in the current state",
            INVALID_FRAMEBUFFER_OPERATION => "The framebuffer object is not complete",
            OUT_OF_MEMORY => "There is not enough memory left to execute the command",
            _ => "Unknown error",
        }
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error({}, \"{}\")", self.0, self.to_str())
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Error({}, \"{}\")", self.0, self.to_str())
    }
}

impl std::error::Error for Error {}
