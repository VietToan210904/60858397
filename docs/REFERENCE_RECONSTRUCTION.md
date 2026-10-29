The additional reference benchmark uses real Google Scanned Objects meshes. It is separate from the original 20 ABO photographs: the GSO inputs are controlled renders of known scanned references, allowing measured shape comparisons. It evaluates the same two pinned SF3D and SPAR3D models and frozen inference wrappers.

The [official Google Research release](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) identifies the scans, their structured-light capture process, CC BY 4.0 licence and hosting through Open Robotics. Each selected model comes from the public `GoogleResearch` owner on Gazebo Fuel. Individual versioned ZIP responses provide version and size; the acquisition manifest records both metadata checks and actual SHA-256. Original OBJ, MTL, texture and copyright metadata are retained unchanged. ZIPs stay in the excluded `.cache` directory. The original MTL's texture basename is bound explicitly to the original texture when creating the normalized GLB.

OmniObject3D was investigated first. Both official author-linked shoe and handbag archives returned Google Drive download-quota errors after ordinary public large-file confirmation. No archive was downloaded and no quota was bypassed. ABO's official mesh-class annotations contained no suitable fashion meshes. The source switch changes the dataset, not either evaluated model.

Selection was fixed before model outputs. From the complete current public owner catalogue, eligible categories are source-labelled shoes and source-labelled bags explicitly named/described as backpacks. Pencil cases and lunch boxes are excluded. Case-sensitive lexical name order selects the first two valid objects per category for development. The test set uses the next 17 shoes and three backpacks; the available five backpacks cannot support ten disjoint test bags. All 24 candidates passed finite geometry, indexed positive surface area, original UV and decoded texture checks, with no exclusions. This creates a pronounced alphabetical/brand and category imbalance; it is a controlled pilot rather than a representative fashion benchmark.

Reference preparation follows Gazebo's [documented +Z-up convention](https://gazebosim.org/api/sim/10/frame_reference.html): fixed Rx(-90 degrees) maps the source to +Y up. The reference alone is centered by its bounding box and scaled to maximum extent one. Exact source transforms are retained per object. The shared conditioning image is a 512-pixel RGBA render from `(0,0,-2.5)`, +Y up, vertical field of view 40 degrees. This is a fixed source-coordinate view, sometimes showing the heel/end of a shoe; it is not claimed to be the semantic front. Both models receive exactly the same input. Four held-out views use azimuths 0/90/180/270 degrees, elevation 20 degrees, distance 2.5 and vertical field of view 45 degrees. Camera poses use OpenGL +X right, +Y up, -Z forward. RGB and binary depth-derived silhouettes are retained.

The machine-readable [protocol](../config/reconstruction_reference_protocol.json) is frozen before development inference. Geometry is reported after independently centering each prediction and reference and giving each a unit bounding-box diagonal. Consequently, these scores measure normalized shape and remove absolute size and translation. The unaligned score retains exported orientation. The additional rigid ICP score explicitly uses the reference to remove nuisance pose and measures shape after alignment. It does not measure absolute reconstruction accuracy.

ICP uses 2,048 separately seeded area-weighted registration samples from each mesh, all 24 proper signed-permutation starting rotations, up to 50 untrimmed nearest-neighbor Kabsch updates, and an absolute improvement tolerance of 1e-7. No reflection or further fitted scale is allowed. The trial with smallest symmetric squared registration Chamfer is selected; lexical order breaks ties. All trial losses, iteration counts and the selected transform are retained. Geometry is then measured on independent 10,000-point samples: symmetric squared Chamfer, unsquared Chamfer and precision/recall/F-score within 1%, 2% and 5% of normalized bounding-box diagonal. Thus measurement points do not determine the fitted pose.

Supplementary silhouette IoU uses the saved reference cameras after the declared normalization, with the rigid-aligned version labelled explicitly. This is a reference-assisted normalized view comparison, not physically calibrated model accuracy. Texture/render similarities and human quality remain distinct. Raw inference timing, VRAM, source/checkpoint identities, failures and input descriptors remain in the separate raw records. The original ABO records and their missing-GT flags are unchanged.

Paired test summaries average objects, report the 17/3 category mix, and use a category-stratified paired object bootstrap (2,000 samples, seed 8414) for model differences. Failures remain in execution coverage; quality summaries are conditional on successful paired objects. Development cases never enter test summaries. Parameters are not chosen from test outputs.

Reproduction uses the existing pinned native model environments and `.venv-preview`. No new model weights or remote services are introduced:

```powershell
python scripts/acquire_gso_reference.py
.venv-preview/Scripts/python.exe scripts/prepare_gso_reference.py
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark freeze
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark run --split development --model sf3d
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark run --split development --model spar3d
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark score --split development
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark authorize-test
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark run --split test --model sf3d
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark run --split test --model spar3d
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark score --split test
.venv-preview/Scripts/python.exe -m stylesphere.reconstruction.reference_benchmark verify
```

The downloader requires the saved complete official catalogue in `.cache/gso_reference/official_models_metadata.json`; a marker ZIP excludes caches. A prepared submission includes the rights-cleared selected originals and derivatives, so acquisition is unnecessary when their frozen hashes verify. The cache catalogue is preserved as a small source metadata snapshot inside the dataset for audit and optional reacquisition.
