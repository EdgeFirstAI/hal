// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod decoder_builder_tests {
    use crate::{
        configs::{self, DecoderType, DecoderVersion, DimName, ModelType, QuantTuple},
        dequantize_ndarray, ConfigOutput, ConfigOutputs, Decoder, DecoderBuilder, DecoderError,
        DetectBox,
    };
    use ndarray::Array3;

    #[test]
    fn test_decoder_builder_no_config() {
        use crate::DecoderBuilder;
        let result = DecoderBuilder::default().build();
        assert!(matches!(result, Err(DecoderError::NoConfig)));
    }

    #[test]
    fn test_decoder_builder_empty_config() {
        use crate::DecoderBuilder;
        let result = DecoderBuilder::default()
            .with_config(ConfigOutputs {
                outputs: vec![],
                ..Default::default()
            })
            .build();
        assert!(
            matches!(result, Err(DecoderError::InvalidConfig(s)) if s == "No outputs found in config")
        );
    }

    #[test]
    fn test_malformed_config_yaml() {
        let malformed_yaml = "
        model_type: yolov8_det
        outputs:
          - shape: [1, 84, 8400]
        "
        .to_owned();
        let result = DecoderBuilder::new()
            .with_config_yaml_str(malformed_yaml)
            .build();
        assert!(matches!(result, Err(DecoderError::Yaml(_))));
    }

    #[test]
    fn test_malformed_config_json() {
        let malformed_yaml = "
        {
            \"model_type\": \"yolov8_det\",
            \"outputs\": [
                {
                    \"shape\": [1, 84, 8400]
                }
            ]
        }"
        .to_owned();
        let result = DecoderBuilder::new()
            .with_config_json_str(malformed_yaml)
            .build();
        assert!(matches!(result, Err(DecoderError::Json(_))));
    }

    #[test]
    fn test_modelpack_and_yolo_config_error() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::ModelPack,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Both ModelPack and Yolo outputs found in config"
        ));
    }

    #[test]
    fn test_yolo_invalid_seg_shape() {
        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400, 1], // Invalid shape
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                    normalized: Some(true),
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")
        ));
    }

    #[test]
    fn test_yolo_invalid_mask() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Mask(configs::Mask {
                    shape: vec![1, 160, 160, 1],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumFeatures, 1),
                    ],
                })],
                ..Default::default()
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Mask output with Yolo decoder")
        ));
    }

    #[test]
    fn test_yolo_invalid_outputs() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Segmentation(configs::Segmentation {
                    shape: vec![1, 84, 8400],
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 84),
                        (DimName::NumBoxes, 8400),
                    ],
                })],
                ..Default::default()
            })
            .build();

        assert!(
            matches!(result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid Segmentation output with Yolo decoder")
        );
    }

    #[test]
    fn test_yolo_invalid_det() {
        let result = DecoderBuilder::new()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400, 1], // Invalid shape
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 84),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 8400, 3], // Invalid shape
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 3),
                    ],
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(
            matches!(
            &result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid shape: Yolo num_features 3 must be greater than 4")),
            "{}",
            result.unwrap_err()
        );

        let result = DecoderBuilder::new()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 3, 8400], // Invalid shape
                    dshape: Vec::new(),
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid shape: Yolo num_features 3 must be greater than 4")));
    }

    #[test]
    fn test_yolo_invalid_segdet() {
        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400, 1], // Invalid shape
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                    normalized: Some(true),
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Detection shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400],
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1], // Invalid shape
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::Batch, 1),
                    ],
                    quantization: None,
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Protos shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 36], // too few classes
                    quantization: None,
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumFeatures, 36),
                    ],
                    normalized: Some(true),
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
                Some(DecoderVersion::Yolo11),
            )
            .build();
        println!("{:?}", result);
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid shape: Yolo num_features 36 must be greater than 36"));
    }

    #[test]
    fn test_yolo_invalid_split_det() {
        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400, 1], // Invalid shape
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 4, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400, 1], // Invalid shape
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                        (DimName::Batch, 1),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400 + 1, 80], // Invalid number of boxes
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumClasses, 80),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Scores num 8401")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 5, 8400], // Invalid boxes dimensions
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 5),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 80, 8400],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("BoxCoords dimension size must be 4")));
    }

    #[test]
    fn test_yolo_invalid_split_segdet() {
        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                        (DimName::Batch, 1),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],

                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                        (DimName::Batch, 1),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                        (DimName::Batch, 1),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Split Mask Coefficients shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160, 1],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::Batch, 1),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo Protos shape")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Scores num 8401")));

        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8401, 32],

                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8401),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 32, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(ref s)) if s.starts_with("Yolo Split Detection Boxes num 8400 incompatible with Mask Coefficients num 8401")));
        let result = DecoderBuilder::new()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 80],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumClasses, 80),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 8400, 32],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::NumProtos, 32),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    shape: vec![1, 31, 160, 160],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 31),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .build();
        println!("{:?}", result);
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(ref s)) if s.starts_with( "Yolo Protos channels 31 incompatible with Mask Coefficients channels 32")));
    }

    #[test]
    fn test_modelpack_invalid_config() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Boxes(configs::Boxes {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 1, 4],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::Padding, 1),
                            (DimName::BoxCoords, 4),
                        ],
                        normalized: Some(true),
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumClasses, 3),
                        ],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumFeatures, 3),
                        ],
                    }),
                ],
                ..Default::default()
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack should not have protos"));

        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Boxes(configs::Boxes {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 1, 4],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::Padding, 1),
                            (DimName::BoxCoords, 4),
                        ],
                        normalized: Some(true),
                    }),
                    ConfigOutput::Scores(configs::Scores {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumClasses, 3),
                        ],
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: configs::DecoderType::ModelPack,
                        shape: vec![1, 8400, 3],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumBoxes, 8400),
                            (DimName::NumProtos, 3),
                        ],
                    }),
                ],
                ..Default::default()
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack should not have mask coefficients"));

        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![ConfigOutput::Boxes(configs::Boxes {
                    decoder: configs::DecoderType::ModelPack,
                    shape: vec![1, 8400, 1, 4],
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 8400),
                        (DimName::Padding, 1),
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
                })],
                ..Default::default()
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Invalid ModelPack model outputs"));
    }

    #[test]
    fn test_modelpack_invalid_det() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Boxes shape")));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                        (DimName::Padding, 1),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Scores shape")));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 2, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::Padding, 2),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "Padding dimension size must be 1"));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 5, 1, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 5),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "BoxCoords dimension size must be 4"));

        let result = DecoderBuilder::new()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 80, 8401],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8401),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Detection Boxes num 8400 incompatible with Scores num 8401"));
    }

    #[test]
    fn test_modelpack_invalid_det_split() {
        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: None,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                    normalized: Some(true),
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 9, 15, 18],
                    anchors: None,
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 9),
                        (DimName::Width, 15),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                    normalized: Some(true),
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection missing anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18],
                anchors: None,
                quantization: None,
                dshape: Vec::new(),
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection missing anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18],
                anchors: Some(vec![]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                    (DimName::NumAnchorsXFeatures, 18),
                ],
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s == "ModelPack Split Detection has zero anchors"));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 18, 1],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                    (DimName::NumAnchorsXFeatures, 18),
                    (DimName::Padding, 1),
                ],
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Split Detection shape")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 15, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumAnchorsXFeatures, 15),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not greater than number of anchors * 5 =")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 15],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: Vec::new(),
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not greater than number of anchors * 5 =")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 16, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumAnchorsXFeatures, 16),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not a multiple of number of anchors")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 17, 30, 16],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: Vec::new(),
                normalized: Some(true),
            }])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("not a multiple of number of anchors")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![configs::Detection {
                decoder: DecoderType::ModelPack,
                shape: vec![1, 18, 17, 30],
                anchors: Some(vec![
                    [0.3666666, 0.3148148],
                    [0.3874999, 0.474074],
                    [0.5333333, 0.644444],
                ]),
                quantization: None,
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 18),
                    (DimName::Height, 17),
                    (DimName::Width, 30),
                ],
                normalized: Some(true),
            }])
            .build();
        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("Split Detection dshape missing required dimension NumAnchorsXFeature")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                    normalized: Some(true),
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 21],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 21),
                    ],
                    normalized: Some(true),
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("ModelPack Split Detection inconsistent number of classes:")));

        let result = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    dshape: vec![],
                    normalized: Some(true),
                },
                configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 21],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    dshape: vec![],
                    normalized: Some(true),
                },
            ])
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("ModelPack Split Detection inconsistent number of classes:")));
    }

    #[test]
    fn test_modelpack_invalid_seg() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_seg(configs::Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: None,
                shape: vec![1, 160, 106, 3, 1],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 106),
                    (DimName::NumClasses, 3),
                    (DimName::Padding, 1),
                ],
            })
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid ModelPack Segmentation shape")));
    }

    #[test]
    fn test_modelpack_invalid_segdet() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_segdet(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 1, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::Padding, 1),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 106),
                        (DimName::NumClasses, 3),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("incompatible with number of classes")));
    }

    #[test]
    fn test_modelpack_invalid_segdet_split() {
        let result = DecoderBuilder::new()
            .with_config_modelpack_segdet_split(
                vec![configs::Detection {
                    decoder: DecoderType::ModelPack,
                    shape: vec![1, 17, 30, 18],
                    anchors: Some(vec![
                        [0.3666666, 0.3148148],
                        [0.3874999, 0.474074],
                        [0.5333333, 0.644444],
                    ]),
                    quantization: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 17),
                        (DimName::Width, 30),
                        (DimName::NumAnchorsXFeatures, 18),
                    ],
                    normalized: Some(true),
                }],
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: None,
                    shape: vec![1, 160, 106, 3],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 106),
                        (DimName::NumClasses, 3),
                    ],
                },
            )
            .build();

        assert!(matches!(
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("incompatible with number of classes")));
    }

    #[test]
    fn test_decode_bad_shapes() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let quant = (0.0040811873, -123);
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8s_80_classes.bin"
        ));
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let out_float: Array3<f32> = dequantize_ndarray(out.view(), quant.into());

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    decoder: DecoderType::Ultralytics,
                    shape: vec![1, 85, 8400],
                    anchors: None,
                    quantization: Some(quant.into()),
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 85),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let result =
            decoder.decode_quantized(&[out.view().into()], &mut output_boxes, &mut output_masks);

        assert!(matches!(
            result, Err(DecoderError::InvalidShape(s)) if s == "Did not find output with shape [1, 85, 8400]"));

        let result = decoder.decode_float(
            &[out_float.view().into_dyn()],
            &mut output_boxes,
            &mut output_masks,
        );

        assert!(matches!(
            result, Err(DecoderError::InvalidShape(s)) if s == "Did not find output with shape [1, 85, 8400]"));
    }

    #[test]
    fn test_config_outputs() {
        let outputs = [
            ConfigOutput::Detection(configs::Detection {
                decoder: configs::DecoderType::Ultralytics,
                anchors: None,
                shape: vec![1, 8400, 85],
                quantization: Some(QuantTuple(0.123, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumFeatures, 85),
                ],
                normalized: Some(true),
            }),
            ConfigOutput::Mask(configs::Mask {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 160, 160, 1],
                quantization: Some(QuantTuple(0.223, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                    (DimName::NumFeatures, 1),
                ],
            }),
            ConfigOutput::Segmentation(configs::Segmentation {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 160, 160, 80],
                quantization: Some(QuantTuple(0.323, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                    (DimName::NumClasses, 80),
                ],
            }),
            ConfigOutput::Scores(configs::Scores {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 80],
                quantization: Some(QuantTuple(0.423, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumClasses, 80),
                ],
            }),
            ConfigOutput::Boxes(configs::Boxes {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 4],
                quantization: Some(QuantTuple(0.523, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::BoxCoords, 4),
                ],
                normalized: Some(true),
            }),
            ConfigOutput::Protos(configs::Protos {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 32, 160, 160],
                quantization: Some(QuantTuple(0.623, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumProtos, 32),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ],
            }),
            ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                decoder: configs::DecoderType::Ultralytics,
                shape: vec![1, 8400, 32],
                quantization: Some(QuantTuple(0.723, 0)),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumBoxes, 8400),
                    (DimName::NumProtos, 32),
                ],
            }),
        ];

        let shapes = outputs.clone().map(|x| x.shape().to_vec());
        assert_eq!(
            shapes,
            [
                vec![1, 8400, 85],
                vec![1, 160, 160, 1],
                vec![1, 160, 160, 80],
                vec![1, 8400, 80],
                vec![1, 8400, 4],
                vec![1, 32, 160, 160],
                vec![1, 8400, 32],
            ]
        );

        let quants: [Option<(f32, i32)>; 7] = outputs.map(|x| x.quantization().map(|q| q.into()));
        assert_eq!(
            quants,
            [
                Some((0.123, 0)),
                Some((0.223, 0)),
                Some((0.323, 0)),
                Some((0.423, 0)),
                Some((0.523, 0)),
                Some((0.623, 0)),
                Some((0.723, 0)),
            ]
        );
    }

    #[test]
    fn test_nms_from_config_yaml() {
        // Test parsing NMS from YAML config
        let yaml_class_agnostic = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
nms: class_agnostic
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_class_agnostic.to_string())
            .build()
            .unwrap();
        assert_eq!(decoder.nms, Some(configs::Nms::ClassAgnostic));

        let yaml_class_aware = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
nms: class_aware
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_class_aware.to_string())
            .build()
            .unwrap();
        assert_eq!(decoder.nms, Some(configs::Nms::ClassAware));

        // Test that config NMS overrides builder NMS
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_class_aware.to_string())
            .with_nms(Some(configs::Nms::ClassAgnostic)) // Builder sets agnostic
            .build()
            .unwrap();
        // Config should override builder
        assert_eq!(decoder.nms, Some(configs::Nms::ClassAware));
    }

    #[test]
    fn test_nms_from_config_json() {
        // Test parsing NMS from JSON config
        let json_class_aware = r#"{
            "outputs": [{
                "decoder": "ultralytics",
                "type": "detection",
                "shape": [1, 84, 8400],
                "dshape": [["batch", 1], ["num_features", 84], ["num_boxes", 8400]]
            }],
            "nms": "class_aware"
        }"#;
        let decoder = DecoderBuilder::new()
            .with_config_json_str(json_class_aware.to_string())
            .build()
            .unwrap();
        assert_eq!(decoder.nms, Some(configs::Nms::ClassAware));
    }

    #[test]
    fn test_nms_missing_from_config_uses_builder_default() {
        // Test that missing NMS in config uses builder default
        let yaml_no_nms = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_no_nms.to_string())
            .build()
            .unwrap();
        // Default builder NMS is ClassAgnostic
        assert_eq!(decoder.nms, Some(configs::Nms::ClassAgnostic));

        // Test with explicit builder NMS
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_no_nms.to_string())
            .with_nms(None) // Explicitly set to None (bypass NMS)
            .build()
            .unwrap();
        assert_eq!(decoder.nms, None);
    }

    #[test]
    fn test_decoder_version_yolo26_end_to_end() {
        // Test that decoder_version: yolo26 creates end-to-end model type
        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 6, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 6]
      - [num_boxes, 8400]
decoder_version: yolo26
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        assert!(matches!(
            decoder.model_type,
            ModelType::YoloEndToEndDet { .. }
        ));

        // Even with NMS set, yolo26 should use end-to-end
        let yaml_with_nms = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 6, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 6]
      - [num_boxes, 8400]
decoder_version: yolo26
nms: class_agnostic
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml_with_nms.to_string())
            .build()
            .unwrap();
        assert!(matches!(
            decoder.model_type,
            ModelType::YoloEndToEndDet { .. }
        ));
    }

    #[test]
    fn test_decoder_version_yolov8_traditional() {
        // Test that decoder_version: yolov8 creates traditional model type
        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
decoder_version: yolov8
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        assert!(matches!(decoder.model_type, ModelType::YoloDet { .. }));
    }

    #[test]
    fn test_decoder_version_all_versions() {
        // Test all supported decoder versions parse correctly
        for version in ["yolov5", "yolov8", "yolo11"] {
            let yaml = format!(
                r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
decoder_version: {}
"#,
                version
            );
            let decoder = DecoderBuilder::new()
                .with_config_yaml_str(yaml)
                .build()
                .unwrap();

            assert!(
                matches!(decoder.model_type, ModelType::YoloDet { .. }),
                "Expected traditional for {}",
                version
            );
        }

        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 6, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 6]
      - [num_boxes, 8400]
decoder_version: yolo26
"#
        .to_string();

        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml)
            .build()
            .unwrap();

        assert!(
            matches!(decoder.model_type, ModelType::YoloEndToEndDet { .. }),
            "Expected end to end for yolo26",
        );
    }

    #[test]
    fn test_decoder_version_json() {
        // Test parsing decoder_version from JSON config
        let json = r#"{
            "outputs": [{
                "decoder": "ultralytics",
                "type": "detection",
                "shape": [1, 6, 8400],
                "dshape": [["batch", 1], ["num_features", 6], ["num_boxes", 8400]]
            }],
            "decoder_version": "yolo26"
        }"#;
        let decoder = DecoderBuilder::new()
            .with_config_json_str(json.to_string())
            .build()
            .unwrap();
        assert!(matches!(
            decoder.model_type,
            ModelType::YoloEndToEndDet { .. }
        ));
    }

    #[test]
    fn test_decoder_version_none_uses_traditional() {
        // Without decoder_version, traditional model type is used
        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        assert!(matches!(decoder.model_type, ModelType::YoloDet { .. }));
    }

    #[test]
    fn test_decoder_version_none_with_nms_none_still_traditional() {
        // Without decoder_version, nms: None now means user handles NMS, not end-to-end
        // This is a behavior change from the previous implementation
        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 84, 8400]
    dshape:
      - [batch, 1]
      - [num_features, 84]
      - [num_boxes, 8400]
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .with_nms(None) // User wants to handle NMS themselves
            .build()
            .unwrap();
        // nms=None with 84 features (80 classes) -> traditional YoloDet (user handles
        // NMS)
        assert!(matches!(decoder.model_type, ModelType::YoloDet { .. }));
    }

    #[test]
    fn test_decoder_heuristic_end_to_end_detection() {
        // models with (batch, num_boxes, num_features) output shape are treated
        // as end-to-end detection
        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 300, 6]
    dshape:
      - [batch, 1]
      - [num_boxes, 300]
      - [num_features, 6]
 
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        // 6 features with (batch, N, features) layout -> end-to-end detection
        assert!(matches!(
            decoder.model_type,
            ModelType::YoloEndToEndDet { .. }
        ));

        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 300, 38]
    dshape:
      - [batch, 1]
      - [num_boxes, 300]
      - [num_features, 38]
  - decoder: ultralytics
    type: protos
    shape: [1, 160, 160, 32]
    dshape:
      - [batch, 1]
      - [height, 160]
      - [width, 160]
      - [num_protos, 32]
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        // 7 features with protos -> end-to-end segmentation detection
        assert!(matches!(
            decoder.model_type,
            ModelType::YoloEndToEndSegDet { .. }
        ));

        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 6, 300]
    dshape:
      - [batch, 1]
      - [num_features, 6]
      - [num_boxes, 300] 
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        // 6 features -> traditional YOLO detection (needs num_classes > 0 for
        // end-to-end)
        assert!(matches!(decoder.model_type, ModelType::YoloDet { .. }));

        let yaml = r#"
outputs:
  - decoder: ultralytics
    type: detection
    shape: [1, 38, 300]
    dshape:
      - [batch, 1]
      - [num_features, 38]
      - [num_boxes, 300]

  - decoder: ultralytics
    type: protos
    shape: [1, 160, 160, 32]
    dshape:
      - [batch, 1]
      - [height, 160]
      - [width, 160]
      - [num_protos, 32]
"#;
        let decoder = DecoderBuilder::new()
            .with_config_yaml_str(yaml.to_string())
            .build()
            .unwrap();
        // 38 features (4+2+32) with protos -> traditional YOLO segmentation detection
        assert!(matches!(decoder.model_type, ModelType::YoloSegDet { .. }));
    }

    #[test]
    fn test_decoder_version_is_end_to_end() {
        assert!(!configs::DecoderVersion::Yolov5.is_end_to_end());
        assert!(!configs::DecoderVersion::Yolov8.is_end_to_end());
        assert!(!configs::DecoderVersion::Yolo11.is_end_to_end());
        assert!(configs::DecoderVersion::Yolo26.is_end_to_end());
    }

    #[test]
    fn test_dshape_dict_format() {
        // Spec produces array-of-single-key-dicts: [{"batch": 1}, {"num_features": 84}]
        let json = r#"{
            "decoder": "ultralytics",
            "shape": [1, 84, 8400],
            "dshape": [{"batch": 1}, {"num_features": 84}, {"num_boxes": 8400}]
        }"#;
        let det: configs::Detection = serde_json::from_str(json).unwrap();
        assert_eq!(det.dshape.len(), 3);
        assert_eq!(det.dshape[0], (configs::DimName::Batch, 1));
        assert_eq!(det.dshape[1], (configs::DimName::NumFeatures, 84));
        assert_eq!(det.dshape[2], (configs::DimName::NumBoxes, 8400));
    }

    #[test]
    fn test_dshape_tuple_format() {
        // Serde native tuple format: [["batch", 1], ["num_features", 84]]
        let json = r#"{
            "decoder": "ultralytics",
            "shape": [1, 84, 8400],
            "dshape": [["batch", 1], ["num_features", 84], ["num_boxes", 8400]]
        }"#;
        let det: configs::Detection = serde_json::from_str(json).unwrap();
        assert_eq!(det.dshape.len(), 3);
        assert_eq!(det.dshape[0], (configs::DimName::Batch, 1));
        assert_eq!(det.dshape[1], (configs::DimName::NumFeatures, 84));
        assert_eq!(det.dshape[2], (configs::DimName::NumBoxes, 8400));
    }

    #[test]
    fn test_dshape_empty_default() {
        // When dshape is omitted entirely, default to empty vec
        let json = r#"{
            "decoder": "ultralytics",
            "shape": [1, 84, 8400]
        }"#;
        let det: configs::Detection = serde_json::from_str(json).unwrap();
        assert!(det.dshape.is_empty());
    }

    #[test]
    fn test_dshape_dict_format_protos() {
        let json = r#"{
            "decoder": "ultralytics",
            "shape": [1, 32, 160, 160],
            "dshape": [{"batch": 1}, {"num_protos": 32}, {"height": 160}, {"width": 160}]
        }"#;
        let protos: configs::Protos = serde_json::from_str(json).unwrap();
        assert_eq!(protos.dshape.len(), 4);
        assert_eq!(protos.dshape[0], (configs::DimName::Batch, 1));
        assert_eq!(protos.dshape[1], (configs::DimName::NumProtos, 32));
    }

    #[test]
    fn test_dshape_dict_format_boxes() {
        let json = r#"{
            "decoder": "ultralytics",
            "shape": [1, 8400, 4],
            "dshape": [{"batch": 1}, {"num_boxes": 8400}, {"box_coords": 4}]
        }"#;
        let boxes: configs::Boxes = serde_json::from_str(json).unwrap();
        assert_eq!(boxes.dshape.len(), 3);
        assert_eq!(boxes.dshape[2], (configs::DimName::BoxCoords, 4));
    }

    // ========================================================================
    // Tests for decode_quantized_proto / decode_float_proto
    // ========================================================================

    /// Build a detection-only decoder (YoloDet) — decode_*_proto returns Ok(None).
    fn build_det_only_decoder() -> Decoder {
        DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    decoder: DecoderType::Ultralytics,
                    shape: vec![1, 84, 8400],
                    anchors: None,
                    quantization: Some((0.004, -123).into()),
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 84),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .with_score_threshold(0.25)
            .with_iou_threshold(0.7)
            .build()
            .unwrap()
    }

    #[test]
    fn test_decode_quantized_proto_returns_none_no_model() {
        // Detection-only decoder has no segmentation model → returns Ok(None)
        let decoder = build_det_only_decoder();
        let data = vec![0i8; 84 * 8400];
        let arr = ndarray::Array3::from_shape_vec((1, 84, 8400), data).unwrap();
        let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(50);
        let result = decoder.decode_quantized_proto(&[arr.view().into()], &mut output_boxes);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "detection-only decoder should return None for proto"
        );
    }

    #[test]
    fn test_decode_float_proto_returns_none_no_model() {
        // Detection-only decoder has no segmentation model → returns Ok(None)
        let decoder = build_det_only_decoder();
        let data = vec![0.0f32; 84 * 8400];
        let arr = ndarray::Array3::from_shape_vec((1, 84, 8400), data).unwrap();
        let mut output_boxes: Vec<DetectBox> = Vec::with_capacity(50);
        let result = decoder.decode_float_proto(&[arr.view().into_dyn()], &mut output_boxes);
        assert!(result.is_ok());
        assert!(
            result.unwrap().is_none(),
            "detection-only decoder should return None for proto"
        );
    }

    #[test]
    fn test_decode_quantized_proto_clears_stale_and_decodes() {
        let decoder = build_det_only_decoder();
        let data = vec![0i8; 84 * 8400];
        let arr = ndarray::Array3::from_shape_vec((1, 84, 8400), data).unwrap();

        // Pre-populate output_boxes with stale data
        let mut output_boxes: Vec<DetectBox> = vec![
            DetectBox {
                bbox: crate::BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 1.0,
                    ymax: 1.0,
                },
                score: 0.99,
                label: 0,
            };
            5
        ];
        assert_eq!(output_boxes.len(), 5);

        let result = decoder.decode_quantized_proto(&[arr.view().into()], &mut output_boxes);
        // For det-only models, decode_proto should return Ok(None) for proto
        // data but still decode detections (clearing stale data first).
        // With all-zero input, no detections pass the score threshold.
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert!(
            output_boxes.len() < 5,
            "decode_quantized_proto should clear stale data: got {} items (was 5)",
            output_boxes.len()
        );
    }

    // ── Proto Extraction Tests ───────────────────────────────────────

    #[test]
    fn test_extract_proto_data_quant_with_cached_model() {
        use crate::yolo::impl_yolo_segdet_quant_proto;
        use crate::{Nms, ProtoData, Quantization, XYWH};

        // Load cached YOLOv8 segmentation model outputs
        let boxes_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes_i8.to_vec()).unwrap();

        let protos_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos_i8.to_vec()).unwrap();

        let quant_boxes = Quantization::new(0.019_484_945, 20);
        let quant_protos = Quantization::new(0.020_889_873, -115);

        let mut output_boxes = Vec::with_capacity(50);
        let proto_data: ProtoData = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            &mut output_boxes,
        );

        // Verify detections are produced
        assert!(
            !output_boxes.is_empty(),
            "Expected detections from cached model outputs"
        );

        // Verify proto data shape
        let protos_shape = match &proto_data.protos {
            crate::ProtoTensor::Quantized { protos, .. } => protos.shape().to_vec(),
            crate::ProtoTensor::Float(arr) => arr.shape().to_vec(),
        };
        assert_eq!(protos_shape, [160, 160, 32], "Proto shape mismatch");

        // Verify mask coefficients: one per detection, each length 32
        assert_eq!(
            proto_data.mask_coefficients.len(),
            output_boxes.len(),
            "mask_coefficients count must match detection count"
        );
        for (i, coeff) in proto_data.mask_coefficients.iter().enumerate() {
            assert_eq!(
                coeff.len(),
                32,
                "Detection {i} has {} coefficients, expected 32",
                coeff.len()
            );
        }

        // Verify proto tensor is quantized variant (input was i8)
        assert!(
            matches!(proto_data.protos, crate::ProtoTensor::Quantized { .. }),
            "Expected Quantized proto tensor for i8 input"
        );
    }

    // ── YOLO 2-Way Split Segmentation Tests ──────────────────────────

    #[test]
    fn test_yolo_segdet_2way_valid_with_dshape() {
        let decoder = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumFeatures, 84),
                            (DimName::NumBoxes, 8400),
                        ],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 8400],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumProtos, 32),
                            (DimName::NumBoxes, 8400),
                        ],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 32],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 160),
                            (DimName::Width, 160),
                            (DimName::NumProtos, 32),
                        ],
                    }),
                ],
                ..Default::default()
            })
            .build();
        assert!(decoder.is_ok(), "Expected valid 2-way split: {decoder:?}");
        let decoder = decoder.unwrap();
        assert!(matches!(
            decoder.model_type(),
            ModelType::YoloSegDet2Way { .. }
        ));
    }

    #[test]
    fn test_yolo_segdet_2way_valid_no_dshape() {
        let decoder = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        quantization: None,
                        dshape: vec![],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 8400],
                        quantization: None,
                        dshape: vec![],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 32],
                        quantization: None,
                        dshape: vec![],
                    }),
                ],
                ..Default::default()
            })
            .build();
        assert!(decoder.is_ok(), "Expected valid 2-way split: {decoder:?}");
    }

    #[test]
    fn test_yolo_segdet_2way_invalid_detection_shape() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84], // 2D — invalid
                        quantization: None,
                        dshape: vec![],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 8400],
                        quantization: None,
                        dshape: vec![],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 32],
                        quantization: None,
                        dshape: vec![],
                    }),
                ],
                ..Default::default()
            })
            .build();
        assert!(matches!(
            result,
            Err(DecoderError::InvalidConfig(s)) if s.starts_with("Invalid Yolo 2-Way Detection shape")
        ));
    }

    #[test]
    fn test_yolo_segdet_2way_num_boxes_mismatch() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        quantization: None,
                        dshape: vec![],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 1000], // mismatched N
                        quantization: None,
                        dshape: vec![],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 32],
                        quantization: None,
                        dshape: vec![],
                    }),
                ],
                ..Default::default()
            })
            .build();
        assert!(matches!(
            result,
            Err(DecoderError::InvalidConfig(s)) if s.contains("num_boxes")
        ));
    }

    #[test]
    fn test_yolo_segdet_2way_proto_channel_mismatch() {
        let result = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        quantization: None,
                        dshape: vec![],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 8400],
                        quantization: None,
                        dshape: vec![],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 16], // 16 != 32
                        quantization: None,
                        dshape: vec![],
                    }),
                ],
                ..Default::default()
            })
            .build();
        assert!(matches!(
            result,
            Err(DecoderError::InvalidConfig(s)) if s.contains("Protos channels")
        ));
    }

    #[test]
    fn test_yolo_segdet_2way_decode_float_roundtrip() {
        // Build a 2-way split decoder: 80 classes, 32 protos
        let decoder = DecoderBuilder::new()
            .with_config(ConfigOutputs {
                outputs: vec![
                    ConfigOutput::Detection(configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumFeatures, 84),
                            (DimName::NumBoxes, 8400),
                        ],
                        normalized: Some(true),
                        anchors: None,
                    }),
                    ConfigOutput::MaskCoefficients(configs::MaskCoefficients {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 32, 8400],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumProtos, 32),
                            (DimName::NumBoxes, 8400),
                        ],
                    }),
                    ConfigOutput::Protos(configs::Protos {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 160, 160, 32],
                        quantization: None,
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 160),
                            (DimName::Width, 160),
                            (DimName::NumProtos, 32),
                        ],
                    }),
                ],
                ..Default::default()
            })
            .with_score_threshold(0.25)
            .with_iou_threshold(0.7)
            .build()
            .unwrap();

        // Use the reference yolov8s detection output, but strip mask_coefs
        // (pretend they come from a separate tensor). Detection = [1,84,8400].
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8s_80_classes.bin"
        ));
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };

        // Dequantize the reference detection output
        let quant = crate::Quantization::new(0.0040811873, -123);
        let mut out_f64 = vec![0.0_f64; 84 * 8400];
        crate::dequantize_cpu(out, quant, &mut out_f64);
        let det_arr = ndarray::Array3::from_shape_vec((1, 84, 8400), out_f64).unwrap();

        // Create synthetic mask_coefs [1,32,8400] and protos [1,160,160,32]
        // (all zeros — no real masks, but the decode path should not crash)
        let mask_coefs = ndarray::Array3::<f64>::zeros((1, 32, 8400));
        let protos = ndarray::Array4::<f64>::zeros((1, 160, 160, 32));

        let outputs = [
            det_arr.view().into_dyn(),
            protos.view().into_dyn(),
            mask_coefs.view().into_dyn(),
        ];
        let outputs: Vec<_> = outputs.iter().map(|x| x.view()).collect();

        let mut output_boxes = Vec::with_capacity(100);
        let mut output_masks = Vec::with_capacity(100);
        let result = decoder.decode_float(&outputs, &mut output_boxes, &mut output_masks);
        assert!(result.is_ok(), "decode_float failed: {result:?}");
        // Should detect boxes (same as reference yolov8s test)
        assert!(!output_boxes.is_empty(), "Expected detections");
    }
}
