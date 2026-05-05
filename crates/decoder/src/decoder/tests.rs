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
        // build() now routes YAML through SchemaV2::parse_yaml which
        // bridges to a serde_json::Value before deserialising as v1
        // ConfigOutputs, so structurally-valid YAML that fails the v1
        // schema check surfaces as a Json variant. Either Yaml (from
        // the initial serde_yaml::from_str step) or Json (from the v1
        // fallback) is acceptable here — the contract is "malformed
        // config errors out".
        assert!(
            matches!(
                result,
                Err(DecoderError::Yaml(_)) | Err(DecoderError::Json(_))
            ),
            "expected Yaml or Json error, got {result:?}"
        );
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("`box_coords` axis must have size 4")));
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
                        (DimName::Padding, 1),
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
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("`padding` axis must have size 1")));

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
            result, Err(DecoderError::InvalidConfig(s)) if s.contains("`box_coords` axis must have size 4")));

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

    /// Larger-scale variant of the full-stack mask-alignment repro
    /// that exercises the non-contiguous `protos.to_shape(...)` path
    /// inside `make_segmentation`. Uses 256-anchor input, 32 protos,
    /// 160×160 proto grid with each anchor writing a unique integer
    /// into proto channel K, where K = anchor index modulo 32. Builds
    /// mask_coefs as a one-hot vector at channel K so the rendered
    /// mask value uniquely identifies which anchor it came from.
    /// A mis-indexed mask would lookup the wrong channel → wrong
    /// rendered value → test failure.
    #[test]
    fn yolo_segdet_combined_tensor_large_scale_non_contiguous_crop() {
        use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

        const NC: usize = 2;
        const NM: usize = 32;
        const N: usize = 256;
        const FEAT: usize = 4 + NC + NM;
        const PH: usize = 160;
        const PW: usize = 160;

        let detection_cfg = configs::Detection {
            decoder: DecoderType::Ultralytics,
            quantization: Some(QuantTuple(1.0, 0)),
            shape: vec![1, FEAT, N],
            // Shape is already canonical [batch, num_features, num_boxes];
            // dshape omitted.
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        };
        let protos_cfg = configs::Protos {
            decoder: DecoderType::Ultralytics,
            quantization: Some(QuantTuple(1.0, 0)),
            shape: vec![1, NM, PH, PW],
            // Physical CHW layout (producer writes channels-outer); HAL
            // permutes to canonical HWC via swap_axes driven by this
            // dshape.
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumProtos, NM),
                (DimName::Height, PH),
                (DimName::Width, PW),
            ],
        };

        let decoder = DecoderBuilder::default()
            .with_score_threshold(0.5)
            .with_iou_threshold(0.99) // disable NMS suppression to keep all survivors identifiable
            .with_nms(Some(configs::Nms::ClassAgnostic))
            .add_output(ConfigOutput::Detection(detection_cfg))
            .add_output(ConfigOutput::Protos(protos_cfg))
            .build()
            .expect("YoloSegDet large-scale decoder must build");

        // Detection tensor (1, FEAT, N): 10 anchors pass threshold.
        // anchor idx 10..20: xywh non-overlapping boxes at different
        // positions; class 0 score 0.9; mask_coefs = one-hot at index
        // `k = idx % NM` with value +10 (strong signal). Protos all
        // zero EXCEPT channel k set to marker value `5.0 + 0.1 * k`
        // so dot(coefs, protos) = 10 * (5.0 + 0.1*k) for the correct
        // anchor, and 0 for any mis-indexed lookup. After sigmoid,
        // correct → ~1 (→ 255 u8); mis-indexed → 0.5 (→ 128 u8).
        let mut det_data = vec![0.0f32; FEAT * N];
        let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
        let n_targets = 10usize;
        let target_start = 10usize;
        for t in 0..n_targets {
            let anchor = target_start + t;
            // Non-overlapping bboxes along x-axis: each covers
            // a narrow vertical strip so they never NMS-suppress.
            let xc = 0.05 + 0.08 * t as f32;
            let yc = 0.5;
            set(&mut det_data, 0, anchor, xc);
            set(&mut det_data, 1, anchor, yc);
            set(&mut det_data, 2, anchor, 0.06);
            set(&mut det_data, 3, anchor, 0.4);
            set(&mut det_data, 4, anchor, 0.9); // class 0
            let k = anchor % NM;
            set(&mut det_data, 4 + NC + k, anchor, 10.0); // one-hot mask coef
        }

        let det_tensor: TensorDyn = {
            let t = Tensor::<f32>::new(&[1, FEAT, N], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&det_data);
            }
            TensorDyn::F32(t)
        };

        // Protos (1, NM, PH, PW) NCHW. Channel k filled with value
        // (5.0 + 0.1*k). Channels that appear in `target_start..`
        // anchor k = anchor % NM mappings are 10..20 → k = 10..20.
        let mut proto_data = vec![0.0f32; NM * PH * PW];
        for k in 0..NM {
            let val = 5.0 + 0.1 * k as f32;
            for i in 0..(PH * PW) {
                proto_data[k * PH * PW + i] = val;
            }
        }
        let protos_tensor: TensorDyn = {
            let t = Tensor::<f32>::new(&[1, NM, PH, PW], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&proto_data);
            }
            TensorDyn::F32(t)
        };

        let inputs: Vec<&TensorDyn> = vec![&det_tensor, &protos_tensor];
        let mut out_boxes: Vec<DetectBox> = Vec::with_capacity(50);
        let mut out_masks: Vec<crate::Segmentation> = Vec::with_capacity(50);
        decoder
            .decode(&inputs, &mut out_boxes, &mut out_masks)
            .expect("YoloSegDet large-scale decode must succeed");
        assert_eq!(
            out_boxes.len(),
            n_targets,
            "all {n_targets} targets should survive; got {}",
            out_boxes.len()
        );

        // For each output, compute the expected mask mean from the
        // anchor-k mapping (inferred from bbox centre xc = 0.05+0.08*t
        // → t = round((xc-0.05)/0.08) → anchor = 10+t → k = anchor%NM).
        // Sigmoid(10 * (5.0 + 0.1*k)) ≈ 1.0 for k ≥ 0 → mask mean ≈ 255.
        // If mask coef looks up the wrong channel (not anchor%NM), the
        // rendered value would be 0 · protos = 0 → sigmoid = 0.5 →
        // mean ≈ 128, far from 255.
        for (b, m) in out_boxes.iter().zip(out_masks.iter()) {
            let cx = (b.bbox.xmin + b.bbox.xmax) * 0.5;
            let mean: f32 = {
                let s = &m.segmentation;
                let total: u32 = s.iter().map(|&v| v as u32).sum();
                total as f32 / s.len() as f32
            };
            assert!(
                mean > 250.0,
                "detection centre {cx:.3}: expected ~255 mask mean (correct mask lookup); \
                 got {mean}. If mean is near 128, the mask coef was sigmoid-of-zero — \
                 indicating the mask row was looked up at the wrong anchor index."
            );
        }
    }

    /// Full-stack reproducer for HAILORT_BUG.md's mask-vs-detection
    /// misalignment claim. Exercises the same config the validator
    /// passes to HAL when running HailoRT → HAL-postproc:
    ///
    ///   { type: detection, shape: [1, 116, 8400] }
    ///   { type: protos,    shape: [1, 32, 160, 160] }   // NCHW
    ///
    /// Input tensors encode anchor-identity in both the mask
    /// coefficients and the proto channel ordering so a mis-indexed
    /// mask would produce rendered values wildly different from the
    /// expected pattern. The test builds `Decoder::decode()` through
    /// the public `TensorDyn` API — same entry point the Python
    /// bindings call.
    #[test]
    fn yolo_segdet_combined_detection_protos_full_stack_mask_alignment() {
        use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

        const NC: usize = 2;
        const NM: usize = 2;
        const N: usize = 5;
        const FEAT: usize = 4 + NC + NM;
        const PH: usize = 8;
        const PW: usize = 8;

        let detection_cfg = configs::Detection {
            decoder: DecoderType::Ultralytics,
            quantization: Some(QuantTuple(1.0, 0)),
            shape: vec![1, FEAT, N],
            // Shape is already canonical [batch, num_features, num_boxes];
            // dshape omitted.
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        };
        let protos_cfg = configs::Protos {
            decoder: DecoderType::Ultralytics,
            quantization: Some(QuantTuple(1.0, 0)),
            shape: vec![1, NM, PH, PW],
            // Physical CHW layout (producer writes channels-outer); HAL
            // permutes to canonical HWC via swap_axes driven by this
            // dshape.
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumProtos, NM),
                (DimName::Height, PH),
                (DimName::Width, PW),
            ],
        };

        let decoder = DecoderBuilder::default()
            .with_score_threshold(0.5)
            .with_iou_threshold(0.5)
            .with_nms(Some(configs::Nms::ClassAgnostic))
            .add_output(ConfigOutput::Detection(detection_cfg))
            .add_output(ConfigOutput::Protos(protos_cfg))
            .build()
            .expect("YoloSegDet detection+protos decoder must build");
        assert!(
            matches!(decoder.model_type, ModelType::YoloSegDet { .. }),
            "expected YoloSegDet model type, got {:?}",
            decoder.model_type
        );

        // Build the (1, 8, 5) detection tensor:
        //   anchor 0: xywh (0.2, 0.2, 0.1, 0.1)  class 0 score 0.95  mask_coefs (+3, +3)
        //   anchor 1: below threshold (filler)
        //   anchor 2: xywh (0.8, 0.8, 0.1, 0.1)  class 0 score 0.85  mask_coefs (-3, -3)
        //   anchor 3: below threshold (filler)
        //   anchor 4: below threshold (filler)
        // Two survivors; mask for a0 should sigmoid to ~0.9975 (→ 254
        // u8), mask for a2 to ~0.0025 (→ 1 u8). ≈250-point gap makes
        // any misalignment trivially detectable.
        let mut det_data = vec![0.0f32; FEAT * N];
        let set = |d: &mut [f32], r: usize, c: usize, v: f32| d[r * N + c] = v;
        set(&mut det_data, 0, 0, 0.2);
        set(&mut det_data, 1, 0, 0.2);
        set(&mut det_data, 2, 0, 0.1);
        set(&mut det_data, 3, 0, 0.1);
        set(&mut det_data, 0, 2, 0.8);
        set(&mut det_data, 1, 2, 0.8);
        set(&mut det_data, 2, 2, 0.1);
        set(&mut det_data, 3, 2, 0.1);
        set(&mut det_data, 4, 0, 0.95); // score
        set(&mut det_data, 4, 2, 0.85);
        set(&mut det_data, 6, 0, 3.0);
        set(&mut det_data, 7, 0, 3.0);
        set(&mut det_data, 6, 2, -3.0);
        set(&mut det_data, 7, 2, -3.0);

        let det_tensor: TensorDyn = {
            let t = Tensor::<f32>::new(&[1, FEAT, N], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&det_data);
            }
            TensorDyn::F32(t)
        };
        // Protos in NCHW (1, 2, 8, 8) all-ones — any proto channel
        // contributes identically, so the sign of mask_coefs dominates
        // the rendered mask pixel value.
        let proto_data = vec![1.0f32; NM * PH * PW];
        let protos_tensor: TensorDyn = {
            let t = Tensor::<f32>::new(&[1, NM, PH, PW], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&proto_data);
            }
            TensorDyn::F32(t)
        };

        let inputs: Vec<&TensorDyn> = vec![&det_tensor, &protos_tensor];
        let mut out_boxes: Vec<DetectBox> = Vec::with_capacity(10);
        let mut out_masks: Vec<crate::Segmentation> = Vec::with_capacity(10);
        decoder
            .decode(&inputs, &mut out_boxes, &mut out_masks)
            .expect("YoloSegDet decode must succeed");
        assert_eq!(
            out_boxes.len(),
            2,
            "two anchors above 0.5 should survive; got {}",
            out_boxes.len()
        );
        assert_eq!(out_masks.len(), out_boxes.len(), "one mask per box");

        for (b, m) in out_boxes.iter().zip(out_masks.iter()) {
            let cx = (b.bbox.xmin + b.bbox.xmax) * 0.5;
            let mean: f32 = {
                let s = &m.segmentation;
                let total: u32 = s.iter().map(|&v| v as u32).sum();
                total as f32 / s.len() as f32
            };
            if cx < 0.3 {
                assert!(
                    mean > 200.0,
                    "anchor-0 detection (cx={cx:.2}) should have high mask mean; got {mean}"
                );
            } else if cx > 0.7 {
                assert!(
                    mean < 50.0,
                    "anchor-2 detection (cx={cx:.2}) should have low mask mean; got {mean}"
                );
            } else {
                panic!("unexpected detection centre {cx:.2}");
            }
        }
    }

    /// f16 twin of `yolo_segdet_combined_detection_protos_full_stack_mask_alignment`.
    /// Proves the Orin TensorRT fp16 engine path: decoder accepts native `F16`
    /// tensors, the `decode_float::<f16>` instantiation handles NMS and
    /// coefficient extraction, and segmentation masks are rendered correctly.
    /// The same 0.95/0.85 scores and +3/−3 mask coefficients as the f32 test
    /// produce the same near-saturating sigmoid outputs (the 10-bit f16
    /// mantissa is lossless for these small magnitudes).
    #[test]
    fn yolo_segdet_combined_detection_protos_f16_full_stack() {
        use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};

        const NC: usize = 2;
        const NM: usize = 2;
        const N: usize = 5;
        const FEAT: usize = 4 + NC + NM;
        const PH: usize = 8;
        const PW: usize = 8;

        let detection_cfg = configs::Detection {
            decoder: DecoderType::Ultralytics,
            quantization: None,
            shape: vec![1, FEAT, N],
            dshape: vec![],
            anchors: None,
            normalized: Some(true),
        };
        let protos_cfg = configs::Protos {
            decoder: DecoderType::Ultralytics,
            quantization: None,
            shape: vec![1, NM, PH, PW],
            // Physical CHW (producer channels-outer); HAL permutes to canonical
            // HWC via the dshape-driven stride permutation introduced in
            // EDGEAI-1288.
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumProtos, NM),
                (DimName::Height, PH),
                (DimName::Width, PW),
            ],
        };

        let decoder = DecoderBuilder::default()
            .with_score_threshold(0.5)
            .with_iou_threshold(0.5)
            .with_nms(Some(configs::Nms::ClassAgnostic))
            .add_output(ConfigOutput::Detection(detection_cfg))
            .add_output(ConfigOutput::Protos(protos_cfg))
            .build()
            .expect("YoloSegDet f16 decoder must build");

        // Build identical detection payload, but as half-precision floats.
        let mut det_data = vec![half::f16::ZERO; FEAT * N];
        let set = |d: &mut [half::f16], r: usize, c: usize, v: f32| {
            d[r * N + c] = half::f16::from_f32(v);
        };
        set(&mut det_data, 0, 0, 0.2);
        set(&mut det_data, 1, 0, 0.2);
        set(&mut det_data, 2, 0, 0.1);
        set(&mut det_data, 3, 0, 0.1);
        set(&mut det_data, 0, 2, 0.8);
        set(&mut det_data, 1, 2, 0.8);
        set(&mut det_data, 2, 2, 0.1);
        set(&mut det_data, 3, 2, 0.1);
        set(&mut det_data, 4, 0, 0.95);
        set(&mut det_data, 4, 2, 0.85);
        set(&mut det_data, 6, 0, 3.0);
        set(&mut det_data, 7, 0, 3.0);
        set(&mut det_data, 6, 2, -3.0);
        set(&mut det_data, 7, 2, -3.0);

        let det_tensor: TensorDyn = {
            let t = Tensor::<half::f16>::new(&[1, FEAT, N], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&det_data);
            }
            TensorDyn::F16(t)
        };

        let proto_data = vec![half::f16::from_f32(1.0); NM * PH * PW];
        let protos_tensor: TensorDyn = {
            let t =
                Tensor::<half::f16>::new(&[1, NM, PH, PW], Some(TensorMemory::Mem), None).unwrap();
            {
                let mut m = t.map().unwrap();
                m.as_mut_slice().copy_from_slice(&proto_data);
            }
            TensorDyn::F16(t)
        };

        let inputs: Vec<&TensorDyn> = vec![&det_tensor, &protos_tensor];

        // Path 1: decode() end-to-end with materialized masks.
        let mut out_boxes: Vec<DetectBox> = Vec::with_capacity(10);
        let mut out_masks: Vec<crate::Segmentation> = Vec::with_capacity(10);
        decoder
            .decode(&inputs, &mut out_boxes, &mut out_masks)
            .expect("f16 YoloSegDet decode must succeed");
        assert_eq!(out_boxes.len(), 2, "two anchors above 0.5 should survive");
        assert_eq!(out_masks.len(), 2);

        for (b, m) in out_boxes.iter().zip(out_masks.iter()) {
            let cx = (b.bbox.xmin + b.bbox.xmax) * 0.5;
            let mean: f32 = {
                let s = &m.segmentation;
                let total: u32 = s.iter().map(|&v| v as u32).sum();
                total as f32 / s.len() as f32
            };
            if cx < 0.3 {
                assert!(mean > 200.0, "anchor-0 f16 mask mean {mean}");
            } else if cx > 0.7 {
                assert!(mean < 50.0, "anchor-2 f16 mask mean {mean}");
            } else {
                panic!("unexpected f16 detection centre {cx:.2}");
            }
        }

        // Path 2: decode_proto() produces TensorDyn::F16 protos (native f16,
        // no widening on the decoder side) with matching F16 mask coefficients.
        let mut out_boxes2: Vec<DetectBox> = Vec::with_capacity(10);
        let proto = decoder
            .decode_proto(&inputs, &mut out_boxes2)
            .expect("f16 decode_proto must succeed")
            .expect("YoloSegDet produces ProtoData");
        assert_eq!(out_boxes2.len(), 2);
        assert_eq!(
            proto.protos.dtype(),
            edgefirst_tensor::DType::F16,
            "f16 inputs should yield TensorDyn::F16 protos, got shape={:?}",
            proto.protos.shape()
        );
        assert_eq!(proto.protos.shape(), &[PH, PW, NM]);
        assert_eq!(
            proto.mask_coefficients.dtype(),
            edgefirst_tensor::DType::F16,
            "f16 input → f16 mask_coefficients"
        );
        assert_eq!(proto.mask_coefficients.shape(), &[out_boxes2.len(), NM]);
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
        // Stale data (5 items) must be cleared. The all-zero quantized input
        // may produce a small number of detections due to dequantization
        // (zero_point=-123 → dequant(0) ≈ 0.49 which can pass threshold).
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
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );

        // Verify detections are produced
        assert!(
            !output_boxes.is_empty(),
            "Expected detections from cached model outputs"
        );

        // Verify proto data shape
        assert_eq!(
            proto_data.protos.shape(),
            &[160, 160, 32],
            "Proto shape mismatch"
        );

        // Verify mask coefficients: shape [num_detections, num_protos],
        // dtype I8 (kept raw with quantization metadata).
        assert_eq!(
            proto_data.mask_coefficients.shape(),
            &[output_boxes.len(), 32],
            "mask_coefficients shape must be [N, num_protos]"
        );
        assert_eq!(
            proto_data.mask_coefficients.dtype(),
            edgefirst_tensor::DType::I8,
            "quantized extraction keeps coefficients as raw I8"
        );
        // Verify coefficients carry quantization metadata.
        let coeff_quant = proto_data
            .mask_coefficients
            .quantization()
            .expect("I8 mask_coefficients must carry quantization metadata");
        assert!(
            coeff_quant.is_per_tensor(),
            "coeff quantization should be per-tensor"
        );

        // Verify proto tensor is I8 (input was i8) with per-tensor quantization.
        assert_eq!(
            proto_data.protos.dtype(),
            edgefirst_tensor::DType::I8,
            "Expected TensorDyn::I8 for quantized model"
        );
        let quant = proto_data
            .protos
            .quantization()
            .expect("quantized proto tensor carries quantization metadata");
        assert!(quant.is_per_tensor(), "quantization should be per-tensor");
    }

    /// f32 sibling of `test_extract_proto_data_quant_with_cached_model`.
    ///
    /// Dequantizes the same cached i8 fixture to f32 using the model's
    /// per-tensor quantization parameters, feeds it through the float
    /// proto-extraction path, and asserts detection count + shape match the
    /// quantized reference exactly. The dequantization is
    /// `(v - zp) * scale` per-element; because both paths run NMS with the
    /// same thresholds on numerically equivalent data, the surviving
    /// detection set is identical.
    ///
    /// Regression anchor: any refactor that silently alters the f32 decode
    /// path (e.g. the planned `ProtoData` → `TensorDyn` migration) fails
    /// the count / shape / coefficient-length assertions before any
    /// downstream mask materialization is touched.
    #[test]
    fn test_extract_proto_data_float_with_cached_model() {
        use crate::yolo::{impl_yolo_segdet_float_proto, impl_yolo_segdet_quant_proto};
        use crate::{Nms, ProtoData, Quantization, XYWH};

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

        // First: run the quantized reference to establish ground truth.
        let mut ref_boxes = Vec::with_capacity(50);
        let _ref_proto: ProtoData = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut ref_boxes,
        );
        assert!(
            !ref_boxes.is_empty(),
            "reference (quant) decode produced no detections — fixture broken"
        );

        // Now dequantize to f32 and run the float path.
        let boxes_f32 =
            boxes.mapv(|v| (v as f32 - quant_boxes.zero_point as f32) * quant_boxes.scale);
        let protos_f32 =
            protos.mapv(|v| (v as f32 - quant_protos.zero_point as f32) * quant_protos.scale);

        let mut output_boxes = Vec::with_capacity(50);
        let proto_data: ProtoData = impl_yolo_segdet_float_proto::<XYWH, f32, f32>(
            boxes_f32.view(),
            protos_f32.view(),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );

        // Same detection count as the quantized reference.
        assert_eq!(
            output_boxes.len(),
            ref_boxes.len(),
            "f32 path produced {} detections; quant reference produced {}",
            output_boxes.len(),
            ref_boxes.len()
        );

        // Shape + coefficient-count invariants.
        assert_eq!(proto_data.protos.shape(), &[160, 160, 32]);
        assert_eq!(
            proto_data.mask_coefficients.shape(),
            &[output_boxes.len(), 32]
        );

        // f32 input → TensorDyn::F32 protos + coefficients (dtype matches).
        assert_eq!(proto_data.protos.dtype(), edgefirst_tensor::DType::F32);
        assert_eq!(
            proto_data.mask_coefficients.dtype(),
            edgefirst_tensor::DType::F32
        );

        // Per-detection box-centre correspondence vs. quantized reference.
        // Order is preserved by NMS (both sorted by score desc), so we can
        // zip directly.
        for (i, (q, f)) in ref_boxes.iter().zip(output_boxes.iter()).enumerate() {
            let dcx = ((q.bbox.xmin + q.bbox.xmax) - (f.bbox.xmin + f.bbox.xmax)).abs() * 0.5;
            let dcy = ((q.bbox.ymin + q.bbox.ymax) - (f.bbox.ymin + f.bbox.ymax)).abs() * 0.5;
            assert!(
                dcx < 5e-3 && dcy < 5e-3,
                "det {i}: f32 centre drift from quant reference ({dcx:.4}, {dcy:.4}) \
                 > 5e-3 — f32 / quantized paths should agree on dequant data"
            );
        }
    }

    /// f16 sibling — same fixture as the f32 test, but coefficients and
    /// protos are narrowed from f32 to f16 before feeding the float decode
    /// path. f16 has a 10-bit mantissa so scores near the NMS threshold can
    /// flip, and post-NMS detection count may differ by ±2 from the
    /// reference. The invariants we assert are:
    ///
    ///   - detection count within ±20% of the quantized reference,
    ///   - per-detection bbox centre drift < 5e-3 for the intersection set,
    ///   - coefficient tensor length equals `num_protos`,
    ///   - `ProtoTensor::Float16` variant is returned.
    #[test]
    fn test_extract_proto_data_f16_with_cached_model() {
        use crate::yolo::{impl_yolo_segdet_float_proto, impl_yolo_segdet_quant_proto};
        use crate::{Nms, ProtoData, Quantization, XYWH};

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

        // Quantized reference.
        let mut ref_boxes = Vec::with_capacity(50);
        let _ref_proto: ProtoData = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut ref_boxes,
        );
        assert!(!ref_boxes.is_empty());

        // Dequantize to f32, then narrow to f16.
        let boxes_f16 = boxes.mapv(|v| {
            half::f16::from_f32((v as f32 - quant_boxes.zero_point as f32) * quant_boxes.scale)
        });
        let protos_f16 = protos.mapv(|v| {
            half::f16::from_f32((v as f32 - quant_protos.zero_point as f32) * quant_protos.scale)
        });

        let mut output_boxes = Vec::with_capacity(50);
        let proto_data: ProtoData = impl_yolo_segdet_float_proto::<XYWH, half::f16, half::f16>(
            boxes_f16.view(),
            protos_f16.view(),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );

        // Detection count within ±20% of quantized reference. Tolerance set
        // generously because scores near the 0.45 threshold can flip with
        // f16's 10-bit mantissa.
        let ref_n = ref_boxes.len();
        let got_n = output_boxes.len();
        assert!(got_n > 0, "f16 path produced zero detections");
        let tol = (ref_n as f32 * 0.2).ceil() as usize;
        assert!(
            got_n.abs_diff(ref_n) <= tol,
            "f16 detection count {got_n} diverged from quant reference {ref_n} by > {tol} (20%)"
        );

        // Proto shape + coefficient shape intact.
        assert_eq!(proto_data.protos.shape(), &[160, 160, 32]);
        assert_eq!(
            proto_data.mask_coefficients.shape(),
            &[output_boxes.len(), 32]
        );

        // f16 input → TensorDyn::F16 for BOTH protos and coefficients
        // (native f16 preserved end-to-end — no widening on the decoder side).
        assert_eq!(proto_data.protos.dtype(), edgefirst_tensor::DType::F16);
        assert_eq!(
            proto_data.mask_coefficients.dtype(),
            edgefirst_tensor::DType::F16
        );

        // Centre-drift on the first min(ref_n, got_n) detections. Order is
        // preserved by NMS when scores don't cross the threshold.
        let n = ref_boxes.len().min(output_boxes.len());
        for (i, (q, f)) in ref_boxes
            .iter()
            .zip(output_boxes.iter())
            .take(n)
            .enumerate()
        {
            let dcx = ((q.bbox.xmin + q.bbox.xmax) - (f.bbox.xmin + f.bbox.xmax)).abs() * 0.5;
            let dcy = ((q.bbox.ymin + q.bbox.ymax) - (f.bbox.ymin + f.bbox.ymax)).abs() * 0.5;
            assert!(
                dcx < 5e-3 && dcy < 5e-3,
                "det {i}: f16 centre drift ({dcx:.4}, {dcy:.4}) > 5e-3 vs. quant reference"
            );
        }
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

    // =========================================================================
    // Schema v2 `with_schema()` builder integration tests
    // =========================================================================

    mod schema_v2_builder {
        use crate::schema::SchemaV2;
        use crate::{DecoderBuilder, DecoderError};

        #[test]
        fn with_schema_flat_v2_builds() {
            // A flat v2 schema (no children) should build a decoder
            // identical to the v1 path.
            let j = r#"{
              "schema_version": 2,
              "decoder_version": "yolov8",
              "outputs": [
                {"name": "boxes", "type": "boxes",
                 "shape": [1, 4, 8400],
                 "dshape": [{"batch":1},{"box_coords":4},{"num_boxes":8400}],
                 "dtype": "int8",
                 "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
                 "decoder": "ultralytics", "encoding": "direct", "normalized": true},
                {"name": "scores", "type": "scores",
                 "shape": [1, 80, 8400],
                 "dshape": [{"batch":1},{"num_classes":80},{"num_boxes":8400}],
                 "dtype": "int8",
                 "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
                 "decoder": "ultralytics", "score_format": "per_class"}
              ]
            }"#;
            let schema = SchemaV2::parse_json(j).unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .build()
                .unwrap();
            assert!(
                decoder.decode_program.is_none(),
                "flat schema should have no merge program"
            );
            assert!(decoder.normalized_boxes() == Some(true));
        }

        #[test]
        fn with_schema_v1_yaml_via_shim_builds() {
            // Legacy v1 testdata parsed through the v2 shim and fed to
            // the new builder should produce a working decoder.
            let yaml = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/yolov8_seg.yaml"
            ));
            let schema = SchemaV2::parse_yaml(yaml).unwrap();
            let decoder = DecoderBuilder::new().with_schema(schema).build().unwrap();
            assert!(decoder.decode_program.is_none());
        }

        #[test]
        fn with_schema_rejects_dfl_flat() {
            let j = r#"{
              "schema_version": 2,
              "decoder_version": "yolov8",
              "outputs": [
                {"name": "boxes", "type": "boxes",
                 "shape": [1, 64, 8400],
                 "dshape": [{"batch":1},{"num_features":64},{"num_boxes":8400}],
                 "dtype": "int8",
                 "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
                 "decoder": "ultralytics", "encoding": "dfl", "normalized": true},
                {"name": "scores", "type": "scores",
                 "shape": [1, 80, 8400],
                 "dtype": "int8",
                 "decoder": "ultralytics", "score_format": "per_class"}
              ]
            }"#;
            let schema = SchemaV2::parse_json(j).unwrap();
            let err = DecoderBuilder::new()
                .with_schema(schema)
                .build()
                .unwrap_err();
            assert!(
                matches!(err, DecoderError::NotSupported(_)),
                "expected NotSupported, got {err:?}"
            );
        }

        #[test]
        fn with_schema_rejects_future_version() {
            let j = r#"{"schema_version": 99, "outputs": []}"#;
            let err = SchemaV2::parse_json(j).unwrap_err();
            assert!(matches!(err, DecoderError::NotSupported(_)));
        }

        /// Construct a TensorDyn with the given values. Helper for the
        /// end-to-end decode test below.
        fn make_i16(shape: &[usize], values: &[i16]) -> edgefirst_tensor::TensorDyn {
            use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
            let t = Tensor::<i16>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            let mut m = t.map().unwrap();
            m.as_mut_slice()[..values.len()].copy_from_slice(values);
            drop(m);
            TensorDyn::I16(t)
        }

        fn make_i8(shape: &[usize], values: &[i8]) -> edgefirst_tensor::TensorDyn {
            use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
            let t = Tensor::<i8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
            let mut m = t.map().unwrap();
            m.as_mut_slice()[..values.len()].copy_from_slice(values);
            drop(m);
            TensorDyn::I8(t)
        }

        #[test]
        fn with_schema_split_decode_end_to_end() {
            // End-to-end: build a decoder from a v2 schema with a
            // channel-sub-split boxes output, run `decode` on synthetic
            // tensors, and verify at least one detection survives NMS
            // given a single-anchor high score.
            //
            // Boxes: xy [1,2,3] i16 + wh [1,2,3] i16 → logical [1,4,3]
            // Scores: [1,2,3] i8
            let j = r#"{
              "schema_version": 2,
              "decoder_version": "yolov8",
              "nms": "class_agnostic",
              "outputs": [
                {"name": "boxes", "type": "boxes",
                 "shape": [1, 4, 3],
                 "dshape": [{"batch":1},{"box_coords":4},{"num_boxes":3}],
                 "decoder": "ultralytics", "encoding": "direct", "normalized": true,
                 "outputs": [
                   {"name": "xy", "type": "boxes_xy",
                    "shape": [1, 2, 3],
                    "dshape": [{"batch":1},{"box_coords":2},{"num_boxes":3}],
                    "dtype": "int16",
                    "quantization": {"scale": 1.0e-3, "zero_point": 0, "dtype": "int16"}},
                   {"name": "wh", "type": "boxes_wh",
                    "shape": [1, 2, 3],
                    "dshape": [{"batch":1},{"box_coords":2},{"num_boxes":3}],
                    "dtype": "int16",
                    "quantization": {"scale": 1.0e-3, "zero_point": 0, "dtype": "int16"}}
                 ]},
                {"name": "scores", "type": "scores",
                 "shape": [1, 2, 3],
                 "dshape": [{"batch":1},{"num_classes":2},{"num_boxes":3}],
                 "dtype": "int8",
                 "quantization": {"scale": 0.008, "zero_point": 0, "dtype": "int8"},
                 "decoder": "ultralytics", "score_format": "per_class"}
              ]
            }"#;
            let schema = SchemaV2::parse_json(j).unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.5)
                .with_iou_threshold(0.5)
                .build()
                .unwrap();
            assert!(decoder.decode_program.is_some());

            // Three anchors, xywh normalized ~[0, 1].
            // xy chan 0 = x-centres, chan 1 = y-centres.
            // Anchor 0: center (0.4, 0.4), size (0.2, 0.2) — valid box
            // Anchor 1: zero — will be filtered
            // Anchor 2: zero — will be filtered
            //
            // xy quant scale 1e-3 → 400 * 1e-3 = 0.4
            let xy = make_i16(
                &[1, 2, 3],
                &[
                    400, 0, 0, // xc per anchor
                    400, 0, 0, // yc per anchor
                ],
            );
            let wh = make_i16(
                &[1, 2, 3],
                &[
                    200, 0, 0, // w per anchor
                    200, 0, 0, // h per anchor
                ],
            );
            // Scores: class 0 for anchor 0 is 125 * 0.008 = 1.0 (high).
            let scores = make_i8(
                &[1, 2, 3],
                &[
                    125, 0, 0, // class 0 per anchor
                    0, 0, 0, // class 1 per anchor
                ],
            );

            let inputs: Vec<&edgefirst_tensor::TensorDyn> = vec![&xy, &wh, &scores];

            // Verify merged tensors first (split from decode so we can
            // isolate merge correctness from kernel correctness).
            let prog = decoder.decode_program.as_ref().unwrap();
            let merged = prog.execute(&inputs).unwrap();
            assert_eq!(merged.len(), 2);
            let merged_boxes = &merged[0];
            let merged_scores = &merged[1];
            assert_eq!(merged_boxes.shape(), &[1, 4, 3]);
            assert_eq!(merged_scores.shape(), &[1, 2, 3]);
            // Anchor 0, channel 0: xc = 0.4
            assert!(
                (merged_boxes[[0, 0, 0]] - 0.4).abs() < 1e-3,
                "xc[0] = {}",
                merged_boxes[[0, 0, 0]]
            );
            assert!(
                (merged_scores[[0, 0, 0]] - 1.0).abs() < 1e-2,
                "score[0][0] = {}",
                merged_scores[[0, 0, 0]]
            );

            // NOTE: impl_yolo_split_float caps output to the caller's
            // vector capacity; a zero-capacity vec drops all boxes.
            let mut boxes = Vec::with_capacity(16);
            let mut masks = Vec::with_capacity(0);
            decoder
                .decode(&inputs, &mut boxes, &mut masks)
                .expect("decode failed");
            assert_eq!(boxes.len(), 1, "expected exactly one surviving anchor");
            let b = &boxes[0];
            // label 0, score ~1.0
            assert_eq!(b.label, 0);
            assert!(b.score > 0.9, "score {:?} should be near 1.0", b.score);
            // xywh(0.4, 0.4, 0.2, 0.2) → xyxy(0.3, 0.3, 0.5, 0.5)
            assert!(
                (b.bbox.xmin - 0.3).abs() < 1e-3 && (b.bbox.xmax - 0.5).abs() < 1e-3,
                "unexpected xmin/xmax: {:?}",
                b
            );
        }

        #[test]
        fn with_schema_real_ara2_int8_dvm() {
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/ara2_int8_edgefirst.json"
            ));
            let schema = SchemaV2::parse_json(json).unwrap();
            schema.validate().unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .build()
                .expect("ARA-2 int8 DVM should build");
            // ARA-2 has a split boxes logical, so a merge program must be
            // attached.
            assert!(
                decoder.decode_program.is_some(),
                "ARA-2 int8 DVM should produce a DecodeProgram (split boxes)"
            );
            assert_eq!(decoder.normalized_boxes(), Some(true));
            assert_eq!(decoder.nms, Some(crate::configs::Nms::ClassAgnostic));
        }

        /// Smoke test: build an ARA-2 decoder from the real DVM metadata
        /// and run `decode` against zero-filled synthetic tensors. The
        /// test verifies the whole pipeline (parse → validate → plan →
        /// dequant → merge → dispatch → NMS) executes without panicking,
        /// which shakes out shape/layout/merge bugs that would otherwise
        /// only surface on real hardware.
        #[test]
        fn with_schema_real_ara2_int8_dvm_decode_smoke() {
            use edgefirst_tensor::{Tensor, TensorDyn, TensorMemory, TensorTrait};
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/ara2_int8_edgefirst.json"
            ));
            let schema = SchemaV2::parse_json(json).unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .with_iou_threshold(0.5)
                .build()
                .unwrap();

            fn zero_tensor_i8(shape: &[usize]) -> TensorDyn {
                let t = Tensor::<i8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
                let _ = t.map().unwrap(); // default-zeroed
                TensorDyn::I8(t)
            }
            fn zero_tensor_u8(shape: &[usize]) -> TensorDyn {
                let t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
                let _ = t.map().unwrap();
                TensorDyn::U8(t)
            }

            // Shapes copied from the real edgefirst.json so binding by
            // shape succeeds.
            let xy = zero_tensor_i8(&[1, 2, 8400, 1]);
            let wh = zero_tensor_i8(&[1, 2, 8400, 1]);
            let scores = zero_tensor_u8(&[1, 80, 8400, 1]);
            let mask_coefs = zero_tensor_i8(&[1, 32, 8400, 1]);
            let protos = zero_tensor_i8(&[1, 32, 160, 160]);
            let inputs: Vec<&TensorDyn> = vec![&xy, &wh, &scores, &mask_coefs, &protos];

            let mut boxes = Vec::with_capacity(16);
            let mut masks = Vec::with_capacity(16);
            // All-zero quantized inputs produce near-zero dequantized
            // scores (well under the 0.25 threshold), so the decoder
            // should complete with no detections — but the pipeline
            // itself must not panic.
            decoder
                .decode(&inputs, &mut boxes, &mut masks)
                .expect("ARA-2 int8 decode on zero tensors");
            assert_eq!(
                boxes.len(),
                0,
                "zero-filled tensors should produce no detections above the 0.25 threshold"
            );
        }

        #[test]
        fn with_schema_real_ara2_int16_dvm() {
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/ara2_int16_edgefirst.json"
            ));
            let schema = SchemaV2::parse_json(json).unwrap();
            schema.validate().unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .build()
                .expect("ARA-2 int16 DVM should build");
            assert!(
                decoder.decode_program.is_some(),
                "ARA-2 int16 DVM should produce a DecodeProgram (split boxes)"
            );
        }

        #[test]
        fn with_schema_split_creates_decode_program() {
            // Channel sub-split boxes (ARA-2 style): logical boxes with
            // two children of shapes [1,3,3] + [1,1,3] → logical [1,4,3].
            // Scores remain flat with shape matching split detection
            // convention.
            let j = r#"{
              "schema_version": 2,
              "decoder_version": "yolov8",
              "outputs": [
                {"name": "boxes", "type": "boxes",
                 "shape": [1, 4, 3],
                 "dshape": [{"batch":1},{"box_coords":4},{"num_boxes":3}],
                 "decoder": "ultralytics", "encoding": "direct", "normalized": true,
                 "outputs": [
                   {"name": "xy", "type": "boxes_xy",
                    "shape": [1, 3, 3],
                    "dshape": [{"batch":1},{"box_coords":3},{"num_boxes":3}],
                    "dtype": "int16",
                    "quantization": {"scale": 3.1e-5, "zero_point": 0, "dtype": "int16"}},
                   {"name": "wh", "type": "boxes_wh",
                    "shape": [1, 1, 3],
                    "dshape": [{"batch":1},{"box_coords":1},{"num_boxes":3}],
                    "dtype": "int16",
                    "quantization": {"scale": 3.2e-5, "zero_point": 0, "dtype": "int16"}}
                 ]},
                {"name": "scores", "type": "scores",
                 "shape": [1, 2, 3],
                 "dshape": [{"batch":1},{"num_classes":2},{"num_boxes":3}],
                 "dtype": "int8",
                 "quantization": {"scale": 0.00392, "zero_point": 0, "dtype": "int8"},
                 "decoder": "ultralytics", "score_format": "per_class"}
              ]
            }"#;
            let schema = SchemaV2::parse_json(j).unwrap();
            let decoder = DecoderBuilder::new().with_schema(schema).build().unwrap();
            assert!(
                decoder.decode_program.is_some(),
                "split schema should produce a DecodeProgram"
            );
        }

        // =====================================================================
        // `with_config_json_str` v2 auto-detect tests (the path used by the
        // Python `Decoder.new_from_json_str` static constructor).
        // =====================================================================

        #[test]
        fn json_str_v2_ara2_int16_builds() {
            // Real ARA-2 int16 metadata fed through the JSON-string path
            // must produce the same merge program the schema path
            // produces. This is the validator's target API: pass the raw
            // edgefirst.json string, get a decoder that handles the
            // xy/wh sub-split natively.
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/ara2_int16_edgefirst.json"
            ));
            let decoder = DecoderBuilder::new()
                .with_config_json_str(json.to_string())
                .with_score_threshold(0.25)
                .build()
                .expect("ARA-2 int16 v2 JSON should build via with_config_json_str");
            assert!(
                decoder.decode_program.is_some(),
                "split boxes must produce a DecodeProgram"
            );
            assert_eq!(decoder.normalized_boxes(), Some(true));
        }

        #[test]
        fn json_str_v2_ara2_int8_builds() {
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/ara2_int8_edgefirst.json"
            ));
            let decoder = DecoderBuilder::new()
                .with_config_json_str(json.to_string())
                .with_score_threshold(0.25)
                .build()
                .expect("ARA-2 int8 v2 JSON should build via with_config_json_str");
            assert!(decoder.decode_program.is_some());
        }

        #[test]
        fn json_str_v1_legacy_still_builds() {
            // Legacy v1 JSON must continue to build via the same path
            // (the auto-detect must not regress v1 callers).
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_split.json"
            ));
            let decoder = DecoderBuilder::new()
                .with_config_json_str(json.to_string())
                .build()
                .expect("legacy v1 JSON must still build");
            assert!(
                decoder.decode_program.is_none(),
                "v1 JSON should not produce a DecodeProgram"
            );
        }

        #[test]
        fn json_str_v2_future_version_rejected() {
            // schema_version above MAX_SUPPORTED_SCHEMA_VERSION must
            // surface as DecoderError::NotSupported rather than a serde
            // error, so callers get a useful upgrade message.
            let json = r#"{"schema_version": 99, "outputs": []}"#;
            let err = DecoderBuilder::new()
                .with_config_json_str(json.to_string())
                .build()
                .expect_err("future schema_version must be rejected");
            assert!(
                matches!(err, DecoderError::NotSupported(_)),
                "expected NotSupported, got {err:?}"
            );
        }

        #[test]
        fn with_schema_real_hailo_yolov8seg_builds_and_reports_dfl_reg_max() {
            // Schema-level e2e: parse the canonical Hailo YOLOv8-seg
            // fixture (strides 8/16/32, DFL boxes, sigmoid-applied
            // scores, mask coefficients per-scale, NHWC protos), build
            // a decoder, and confirm:
            //   * a DecodeProgram is attached (split schema)
            //   * normalized == false (pixel coords per HAILORT spec)
            //   * the DFL reg_max extracted from the program == 16
            //     (= 64 feature channels ÷ 4)
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/hailo_yolov8seg_edgefirst.json"
            ));
            let schema = SchemaV2::parse_json(json).unwrap();
            schema.validate().unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .build()
                .expect("Hailo YOLOv8-seg schema should build");
            assert!(
                decoder.decode_program.is_some(),
                "Hailo schema has per-scale children; DecodeProgram required"
            );
            assert_eq!(decoder.normalized_boxes(), Some(false));
            assert_eq!(
                decoder.decode_program.as_ref().unwrap().boxes_reg_max(),
                Some(16),
                "reg_max must be 16 (64 feature channels ÷ 4)"
            );
        }

        /// End-to-end numerical parity with the validator's synthetic
        /// test (`scripts/decode_hailo_split.py::test_synthetic`):
        /// feed uniform uint8=128 tensors through the whole pipeline
        /// and confirm the DFL-decoded first anchor's xc / w match the
        /// analytic values from HAILORT_DECODER.md §"Test Vectors".
        ///
        /// Uniform 128 produces uniform DFL logits (scale-independent
        /// post-dequant since softmax normalises), so every anchor
        /// decodes to distance `(reg_max-1)/2 = 7.5` on all four sides.
        /// For the first stride-8 anchor at grid (0.5, 0.5):
        ///   xc = (0.5 + 0) * 8 = 4.0
        ///   w  = (7.5 + 7.5) * 8 = 120.0
        #[test]
        fn hailo_yolov8seg_uniform_uint8_128_dfl_decode_parity() {
            use edgefirst_tensor::{Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait};
            let json = include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/hailo_yolov8seg_edgefirst.json"
            ));
            let schema = SchemaV2::parse_json(json).unwrap();
            let decoder = DecoderBuilder::new()
                .with_schema(schema)
                .with_score_threshold(0.25)
                .with_iou_threshold(0.5)
                .build()
                .unwrap();

            fn uniform_u8(shape: &[usize], value: u8) -> TensorDyn {
                let t = Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None).unwrap();
                {
                    let mut m = t.map().unwrap();
                    for byte in m.as_mut_slice() {
                        *byte = value;
                    }
                }
                TensorDyn::U8(t)
            }

            // Shapes in schema order. 128 across the board → uniform
            // DFL distributions and sigmoid-pre-applied scores of
            // `128 × (1/255) ≈ 0.502`.
            let b0 = uniform_u8(&[1, 80, 80, 64], 128);
            let b1 = uniform_u8(&[1, 40, 40, 64], 128);
            let b2 = uniform_u8(&[1, 20, 20, 64], 128);
            let s0 = uniform_u8(&[1, 80, 80, 80], 128);
            let s1 = uniform_u8(&[1, 40, 40, 80], 128);
            let s2 = uniform_u8(&[1, 20, 20, 80], 128);
            let m0 = uniform_u8(&[1, 80, 80, 32], 128);
            let m1 = uniform_u8(&[1, 40, 40, 32], 128);
            let m2 = uniform_u8(&[1, 20, 20, 32], 128);
            let protos = uniform_u8(&[1, 160, 160, 32], 128);
            let inputs: Vec<&TensorDyn> =
                vec![&b0, &b1, &b2, &s0, &s1, &s2, &m0, &m1, &m2, &protos];

            // Reach into the decode program directly to verify the
            // post-DFL merged boxes tensor rather than running the full
            // NMS — uniform input produces 8400 near-identical
            // candidates which collapse arbitrarily through NMS.
            let program = decoder
                .decode_program
                .as_ref()
                .expect("Hailo schema compiles a DecodeProgram");
            let merged = program.execute(&inputs).unwrap();

            // Schema order: boxes, scores, mask_coefs, protos.
            let boxes = &merged[0];
            assert_eq!(
                boxes.shape(),
                &[1, 4, 8400],
                "post-DFL merged boxes must be (1, 4, 8400)"
            );

            // First anchor = stride-8 grid (0, 0) → xc = 4.0, w = 120.
            assert!(
                (boxes[[0, 0, 0]] - 4.0).abs() < 1e-2,
                "first anchor xc = {}, expected ~4.0",
                boxes[[0, 0, 0]]
            );
            assert!(
                (boxes[[0, 1, 0]] - 4.0).abs() < 1e-2,
                "first anchor yc = {}, expected ~4.0",
                boxes[[0, 1, 0]]
            );
            assert!(
                (boxes[[0, 2, 0]] - 120.0).abs() < 1e-1,
                "first anchor w = {}, expected ~120.0",
                boxes[[0, 2, 0]]
            );
            assert!(
                (boxes[[0, 3, 0]] - 120.0).abs() < 1e-1,
                "first anchor h = {}, expected ~120.0",
                boxes[[0, 3, 0]]
            );

            // Scores merged: (1, 80, 8400). Sigmoid-pre-applied with
            // scale=1/255 and zp=0 ⇒ dequant(128) = 128/255 ≈ 0.5020.
            let scores = &merged[1];
            assert_eq!(scores.shape(), &[1, 80, 8400]);
            let s00 = scores[[0, 0, 0]];
            assert!(
                (s00 - 0.5020).abs() < 1e-3,
                "score[0,0,0] = {s00}, expected ~0.5020"
            );

            // Mask coefs merged: (1, 32, 8400).
            let mask_coefs = &merged[2];
            assert_eq!(mask_coefs.shape(), &[1, 32, 8400]);

            // Protos passthrough: (1, 160, 160, 32) — schema declares
            // NHWC; HAL does not transpose on merge. Downstream mask
            // rendering is a separate concern (NHWC_PROTOS.md).
            let protos_out = &merged[3];
            assert_eq!(protos_out.shape(), &[1, 160, 160, 32]);
        }
    }

    // =========================================================================
    // Legacy ConfigOutput serde tag vocabulary
    // =========================================================================

    mod config_output_serde {
        use crate::ConfigOutput;

        #[test]
        fn mask_coefs_is_primary_spelling() {
            // v2 spec vocabulary uses `mask_coefs`. The legacy ConfigOutput
            // enum must accept it so that any consumer re-serialising through
            // the legacy path (or feeding a v2-vocabulary dict into the
            // legacy deserialiser) stays compatible.
            let j = r#"{"type": "mask_coefs", "shape": [1, 32, 8400]}"#;
            let parsed: ConfigOutput = serde_json::from_str(j).unwrap();
            assert!(matches!(parsed, ConfigOutput::MaskCoefficients(_)));
        }

        #[test]
        fn mask_coefficients_alias_still_accepted() {
            // Legacy v1 spelling — required for backward compatibility with
            // models already trained and stored before the v2 vocabulary
            // landed.
            let j = r#"{"type": "mask_coefficients", "shape": [1, 32, 8400]}"#;
            let parsed: ConfigOutput = serde_json::from_str(j).unwrap();
            assert!(matches!(parsed, ConfigOutput::MaskCoefficients(_)));
        }
    }

    /// Verify that a transposed `[K, H, W]` proto array (viewed as `[H, W, K]`
    /// via `permuted_axes`) is detected as NCHW layout and copied correctly
    /// without transposing.
    #[test]
    fn test_extract_proto_nchw_layout_detection() {
        use crate::yolo::impl_yolo_segdet_quant_proto;
        use crate::{Nms, ProtoLayout, Quantization, XYWH};

        let boxes_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let boxes_i8 =
            unsafe { std::slice::from_raw_parts(boxes_raw.as_ptr() as *const i8, boxes_raw.len()) };
        let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes_i8.to_vec()).unwrap();

        // Create protos in physical NCHW layout [K, H, W] then view as [H, W, K].
        let (k, h, w) = (32, 160, 160);
        let protos_raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let protos_i8 = unsafe {
            std::slice::from_raw_parts(protos_raw.as_ptr() as *const i8, protos_raw.len())
        };
        // Create as [K, H, W] physical layout, then permute axes to [H, W, K].
        // This mimics what DMA-BUF model outputs look like when the model
        // produces NCHW protos that are axis-swapped at the ndarray level.
        let protos_nchw = ndarray::Array3::from_shape_vec((k, h, w), protos_i8.to_vec()).unwrap();
        let protos_hwk = protos_nchw.view().permuted_axes([1, 2, 0]);

        // Verify the strides match our NCHW detection pattern [w, 1, h*w].
        assert_eq!(
            protos_hwk.strides(),
            &[w as isize, 1, (h * w) as isize],
            "permuted view should have NCHW strides"
        );

        let quant_boxes = Quantization::new(0.019_484_945, 20);
        let quant_protos = Quantization::new(0.020_889_873, -115);

        let mut output_boxes = Vec::with_capacity(50);
        let proto_data = impl_yolo_segdet_quant_proto::<XYWH, _, _>(
            (boxes.view(), quant_boxes),
            (protos_hwk, quant_protos),
            0.45,
            0.45,
            Some(Nms::ClassAgnostic),
            crate::yolo::MAX_NMS_CANDIDATES,
            300,
            &mut output_boxes,
        );

        // NCHW layout should be detected.
        assert_eq!(
            proto_data.layout,
            ProtoLayout::Nchw,
            "Expected NCHW layout detection for transposed view"
        );

        // Proto tensor shape should be [K, H, W] (NCHW).
        assert_eq!(
            proto_data.protos.shape(),
            &[k, h, w],
            "NCHW proto shape should be [K, H, W]"
        );

        // Verify detections still produced (same model data).
        assert!(
            !output_boxes.is_empty(),
            "Expected detections from NCHW proto path"
        );

        // Verify mask coefficients shape.
        assert_eq!(
            proto_data.mask_coefficients.shape(),
            &[output_boxes.len(), k],
        );
    }
}
