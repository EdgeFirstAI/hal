// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Build a [`Decoder`] from the schema-v2 `edgefirst.json` that ModelPack
//! embeds for a multitask (detection + segmentation) model.
//!
//! The schema lists, in order: per-FPN-scale `type: detection` anchor grids,
//! a raw `type: segmentation` per-class map (canonical NHWC `[1, H, W,
//! num_classes]`), and a decoded `type: masks` argmax class-index map
//! (`[1, H, W]`). The model builds as `ModelPackSegDetSplit`: the `masks`
//! head is not read, and the segmentation class count is taken from the NHWC
//! channel axis.

use edgefirst_decoder::{schema::SchemaV2, DecoderBuilder};

/// A 2-scale multitask ModelPack model with 2 annotated classes.
/// Detection: 3 anchors × (5 + 2 classes) = 21 channels per grid cell.
/// Segmentation: 2 classes + background = 3 channels (NHWC).
const MODELPACK_MULTITASK_JSON: &str = r#"
{
  "schema_version": 2,
  "input": { "shape": [1, 3, 480, 640] },
  "dataset": { "classes": ["cup", "saucer"] },
  "outputs": [
    {
      "name": "output_0", "type": "detection", "encoding": "anchor",
      "decoder": "modelpack", "normalized": true,
      "shape": [1, 30, 40, 21],
      "dshape": [{"batch": 1}, {"height": 30}, {"width": 40},
                 {"num_anchors_x_features": 21}],
      "stride": [16, 16],
      "anchors": [[0.05, 0.05], [0.1, 0.1], [0.2, 0.2]],
      "dtype": "float32", "quantization": null
    },
    {
      "name": "output_1", "type": "detection", "encoding": "anchor",
      "decoder": "modelpack", "normalized": true,
      "shape": [1, 15, 20, 21],
      "dshape": [{"batch": 1}, {"height": 15}, {"width": 20},
                 {"num_anchors_x_features": 21}],
      "stride": [32, 32],
      "anchors": [[0.3, 0.3], [0.5, 0.5], [0.8, 0.8]],
      "dtype": "float32", "quantization": null
    },
    {
      "name": "output_seg", "type": "segmentation", "decoder": "modelpack",
      "shape": [1, 480, 640, 3],
      "dshape": [{"batch": 1}, {"height": 480}, {"width": 640},
                 {"num_classes": 3}],
      "dtype": "float32", "quantization": null
    },
    {
      "name": "output_seg_decoded", "type": "masks", "decoder": "modelpack",
      "shape": [1, 480, 640],
      "dshape": [{"batch": 1}, {"height": 480}, {"width": 640}],
      "dtype": "int32", "quantization": null
    }
  ]
}
"#;

#[test]
fn build_decoder_from_modelpack_multitask_schema() {
    let schema = SchemaV2::parse_json(MODELPACK_MULTITASK_JSON)
        .expect("parse ModelPack multitask edgefirst.json");
    DecoderBuilder::new()
        .with_schema(schema)
        .build()
        .expect("build ModelPack multitask decoder");
}

#[test]
fn parse_tolerates_unknown_input_dshape_axis_name() {
    // An input dshape whose channel axis is named `channels`, which `DimName`
    // does not define.
    let json = MODELPACK_MULTITASK_JSON.replace(
        "\"input\": { \"shape\": [1, 3, 480, 640] }",
        "\"input\": { \"shape\": [1, 480, 640, 3], \"dshape\": [\
         {\"batch\": 1}, {\"height\": 480}, {\"width\": 640}, {\"channels\": 3}] }",
    );
    let schema =
        SchemaV2::parse_json(&json).expect("parse metadata with unknown input dshape axis name");
    DecoderBuilder::new()
        .with_schema(schema)
        .build()
        .expect("build decoder despite unknown input dshape axis name");
}
