# ACSim-GPU
# Sonar Ray Tracing Workflow

This project uses Blender for scene preparation and a Python script for sonar ray tracing.

## Overview

The workflow is:

1. Prepare and process your 3D model in Blender
2. Load the script `sonarRT_UIpanels.py` in Blender
3. Define or assign materials in Blender
4. Run the processing step to save the Blender scene contents
5. Run `ray_tracing_sonar.py` to perform sonar ray tracing

---

## 1. Prepare the model in Blender

First, open your model in Blender and make sure the geometry is ready for processing.

Recommended checks:

- Apply object transforms if necessary
- Make sure the mesh is clean
- Confirm the object orientation and scale are correct
- Remove unnecessary objects from the scene

---

## 2. Load `sonarRT_UIpanels.py`

In Blender, load and run the script:

`sonarRT_UIpanels.py`

This script provides the UI panel used for processing the scene.  
The material settings used by the sonar renderer are also defined in this script.

Please make sure the required materials are properly assigned before continuing.

---

## 3. Define materials

The sonar-related material parameters are defined in `sonarRT_UIpanels.py`.

In Blender, assign the correct materials to the objects in your scene before running the processing step.

---

## 4. Run the processing step

After the scene and materials are ready, run the processing function from the Blender UI panel.

This step saves the necessary Blender scene contents so they can be used later by the ray tracing script.

In other words, this step exports the processed scene data from Blender for sonar rendering.

---

## 5. Run `ray_tracing_sonar.py`

After the Blender scene has been processed and saved, run:

```bash
python ray_tracing_sonar.py
