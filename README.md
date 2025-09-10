# CitySketch (MetaCityLab Project): Structure-First GNN for Hand-Drawn City Maps (CityCog-GNNs)

This project contains a **complete** pipeline:

- **Data ingestion** from shapefiles & relations
- **Structure Skeleton Graph (SSG)** construction with invariance
- **Topology & shape descriptors**
- **Unsupervised training**
- **Supervised / semi-supervised training** 
- **Group-vs-Group similarity**
- **Accuracy vs RealMap** 
- **Score calibration** for Fidelity


The data input is supported by our webGIS-based City Image collection platform:
- The platform can be accessed via MetaCityLab.com.
- Since the domain name may be restricted in certain international regions, if the main domain is inaccessible, it is recommended to access via the IP address.
- IP addresses:
- http://106.15.206.244/
 (MetaCityLab website) server located in Shanghai
- http://43.100.135.195/
 (City Image collection platform) server located in Hong Kong
- Use VPN if needed


Notice

- We are happy to share our platform and code.
- Please note that our project uses paid APIs (e.g., Google Maps API), which come with token limits.
- Commercial use or unrestricted large-scale access may cause significant issues for the availability of our project.
- Therefore, please do not use our shared resources for commercial purposes or distribute them without restrictions.
- For large-scale usage needs, please contact us directly or leave a message on our website — we will be glad to provide assistance.

## Install
```bash
pip install -r requirements.txt
# For torch-geometric, follow their platform-specific instructions if pip fails:
# https://pytorch-geometric.readthedocs.io/
```
