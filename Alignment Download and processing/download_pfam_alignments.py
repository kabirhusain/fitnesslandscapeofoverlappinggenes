import os
import requests

# List of PFam IDs to download
# pfam_ids = [
#     "PF00004", "PF00005", "PF00041", "PF00072", "PF00076", "PF00096",
#     "PF00153", "PF00271", "PF00397", "PF00512", "PF00595", "PF02518", "PF07679"
# ]

pfam_ids = ["PF00009", "PF00011", "PF00013", "PF00012", "PF00016", "PF00017", "PF00018"]

# Base URL for PFam alignments (updated for InterPro)
base_url = "https://www.ebi.ac.uk/interpro/api/protein/pfam/{}/alignment/full"

# Directory to save the alignments
output_dir = "../Data/20250527_Pfam_alignments"
os.makedirs(output_dir, exist_ok=True)

# Download each alignment
for pfam_id in pfam_ids:
    url = base_url.format(pfam_id)
    output_file = os.path.join(output_dir, f"{pfam_id}.aln")
    print(f"Downloading {pfam_id} alignment...")
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, verify=False)  # Disable SSL verification
    if response.status_code == 200:
        with open(output_file, "w") as f:
            f.write(response.text)
        print(f"Saved {pfam_id} alignment to {output_file}")
    else:
        print(f"Failed to download {pfam_id}: HTTP {response.status_code}")
