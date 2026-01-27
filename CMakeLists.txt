cmake_minimum_required(VERSION 3.5)

project(ImpactDoseAcc)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/vboussot/SlicerImpactDoseAcc")
set(EXTENSION_CATEGORY "Radiotherapy")
set(EXTENSION_CONTRIBUTORS "Cédric Hémon (University of Rennes, France), Valentin Boussot (University of Rennes, France), Jean-Louis Dillenseger (University of Rennes, France)")
set(EXTENSION_DESCRIPTION "End-to-end uncertainty estimation framework for dose accumulation with uncertainty quantification")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/vboussot/SlicerImpactDoseAcc/main/ImpactDoseAcc.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/vboussot/SlicerImpactDoseAcc/main/docs/delivered_dose.png https://raw.githubusercontent.com/vboussot/SlicerImpactDoseAcc/main/docs/dose_accumulation.png https://raw.githubusercontent.com/vboussot/SlicerImpactDoseAcc/main/docs/QA.png https://raw.githubusercontent.com/vboussot/SlicerImpactDoseAcc/main/docs/DVH.png")
set(EXTENSION_DEPENDS "SlicerRT")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(ImpactDoseAcc)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

