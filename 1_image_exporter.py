# coding: utf-8
import arcpy
import os

# Pseudocode
# 1. Use an ArcGIS Pro Project with a template that includes:
#     - NAIP Imagery Layer
#     - CAFO Sites Layer
#     - Borderless layout from map
# 2. Set references to:
#     - Project
#     - map
#     - cafos_lyr
#     - layout
#     - Extents needed (python list?)
#     - Output images directory
# 3. Iterate on each cafos_lyr record. In the iteration, perform the following:
#     - Change the extent of the layout to the current site.
#     - For each scale needed:
#         - Set the camera scale to the scale
#         - Export to JPEG to the output images directory using formatting for naming

# aprx = arcpy.mp.ArcGISProject("CURRENT")
aprx = arcpy.mp.ArcGISProject(r"")  # Set path to your ArcGIS Pro Project

m = aprx.listMaps()[0]
cafos_lyr = m.listLayers("ky_afo_lonx_nonzero")[0]
cafos_lyr_id_field = "FID"
lyt = aprx.listLayouts()[0]
mf = lyt.listElements()[0]
output_dir = r""  # Set path to your output folder
cafos_scales_list = [1000, 2000, 3000]

# cafos_lyr_name = arcpy.GetParameterAsText(0)
# cafos_lyr_id_field = arcpy.GetParameterAsText(1)


def main(
    aprx_obj,
    map_obj,
    lyr_obj,
    lyr_id_field,
    layout_obj,
    mapFrame_obj,
    output_image_dir,
    scales_list=[1000],
    projection_wkid="3857"):

    feature_count = arcpy.GetCount_management(lyr_obj).getOutput(0)

    arcpy.AddMessage("Extracting {0} images at specified scales for object labeling...".format(feature_count))

    counter = 0

    with arcpy.da.SearchCursor(lyr_obj, ['SHAPE@', lyr_id_field]) as cursor:
        for row in cursor:
            counter+=1
            arcpy.AddMessage("Extracting image {0} of {1}...".format(counter, feature_count))

            extent = row[0].extent.projectAs(projection_wkid)

            mapFrame_obj.camera.setExtent(extent)

            for scale in scales_list:
                mapFrame_obj.camera.scale = scale

                image_name = "img_{0}_{1}".format(str(row[1]), str(scale))
                layout_obj.exportToJPEG(os.path.join(output_image_dir, image_name))

    arcpy.AddMessage("Extraction completed.")

main(aprx, m, cafos_lyr, cafos_lyr_id_field, lyt, mf, output_dir, cafos_scales_list)
