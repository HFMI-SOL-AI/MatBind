import io
import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import plotly.graph_objects as go
from matminer.featurizers.structure import XRDPowderPattern
from mp_api.client import MPRester
from PIL import Image
from pymatgen.analysis.diffraction.tem import TEMCalculator
from pymatgen.electronic_structure.core import Spin
from robocrys import StructureCondenser, StructureDescriber
from tqdm import tqdm

API_KEY = ""
featurizer = XRDPowderPattern()


def save_data(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def load_data(filename):
    try:
        with open(filename) as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        return []

def fetch_dos_data(material_id):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    # material_id, band_gap = pair
    try:
        # Create a new MPRester object for each thread/process
        with MPRester(API_KEY) as mpr:
            # Query the Density of States for the material
            dos = mpr.get_dos_by_material_id(material_id)

            structure = mpr.get_structure_by_material_id(material_id)
            xrd = featurizer.featurize(structure)

            # # Extract the total density of states and corresponding energies
            total_dos = dos.get_densities(Spin.up) if Spin.up in dos.densities else dos.get_densities()
            energies = dos.energies
            efermi = dos.efermi

            condensed_structure = condenser.condense_structure(structure)
            structure_description = describer.describe(condensed_structure)

            # Return the data as a dictionary
            return {
                "material_id": material_id,
                "efermi": efermi,
                # "band_gap" : band_gap,
                "crystal_structure": structure.as_dict(),
                "xrd": xrd.tolist(),
                "energies": energies.tolist(),
                "total_dos": total_dos.tolist(),
                "structure_description": structure_description
            }
    except Exception as e:
        print(f"Failed to retrieve info for material {material_id}: {e}")
        return None

def new_loading():
    output_file = "/p/project1/solai/yang21/hackathon2024/scripts/resource/four_modality_mp.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_data = load_data(output_file)

    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(fields=["material_id"])
        material_ids = {doc.material_id for doc in results}
        # print(f"Found {len(material_ids)} perovskite materials.")

        processed_ids = {entry["material_id"] for entry in all_data}
        new_material_ids = list(material_ids - processed_ids)

    print(len(material_ids))
    print(len(new_material_ids))
    with open(output_file, "a") as f:
    # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
        # Submit tasks and retrieve results in parallel
            for result in executor.map(fetch_dos_data, new_material_ids):
                if result is not None:
                    # Save each result as a JSON line
                    f.write(json.dumps(result) + "\n")
                    f.flush()  # Ensure data is written immediately
    # save_data(dos_data, "/p/project1/solai/yang21/hackathon2024/scripts/resource/four_modality_mp.json")
    # return dos_data

def main():
    # Load previously fetched data if exists
    all_data = load_data("/p/project1/solai/yang21/hackathon2024/scripts/resource/four_modality_mp.json")

    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(fields=["material_id"])
        material_ids = {doc.material_id for doc in results}
        print(f"Found {len(material_ids)} perovskite materials.")

        processed_ids = {entry["material_id"] for entry in all_data}
        new_material_ids = list(material_ids - processed_ids)

    # Set up multiprocessing
    num_cores = mp.cpu_count()
    pool = mp.Pool(processes=num_cores)

    # Process materials in parallel
    for result in tqdm(pool.imap_unordered(fetch_dos_data, new_material_ids), total=len(new_material_ids), desc="Fetching data"):
        all_data.append(result)
        # Save after processing each material to minimize loss in case of interruption
        save_data(all_data, "/p/project/solai/yang21/hackathon2024/scripts/resource/four_modality_mp.json")

    pool.close()
    pool.join()

    print("Data fetching and serialization complete.")

def get_perovskite_id():
    with MPRester(API_KEY) as mpr:
        results = mpr.materials.summary.search(formula="ABC3", fields=["material_id"])
    material_ids = [doc.material_id for doc in results]
    # print(len(material_ids))
    save_data(material_ids, "/p/project/solai/yang21/hackathon2024/scripts/resource/perovskite_material_id.json")

def generate_diffraction_from_sturcture():
    with MPRester(api_key=API_KEY) as mpr:
    # first retrieve the relevant structure
        structure = mpr.get_structure_by_material_id("mp-1183063")

    # this example shows how to obtain an XRD diffraction pattern
    # these patterns are calculated on-the-fly from the structure
    calculator = TEMCalculator(camera_length=50)
    points = calculator.generate_points(-10, 11)
    tem_dots = calculator.tem_dots(structure, points)
    xs = []
    ys = []
    hkls = []
    intensities = []
    for dot in tem_dots:
        if dot.hkl != (0, 0, 0):
            xs.append(dot.position[0])
            ys.append(dot.position[1])
            hkls.append(str(dot.hkl))
            intensities.append(dot.intensity)
    data = [
        go.Scatter(
            x=xs,
            y=ys,
            text=hkls,
            hoverinfo="skip",
            mode="markers",
            marker={
                "size": 4,
                "cmax": 0,
                "cmin": -5,
                "color": np.log(intensities),
                "colorscale": "Greys",
            },
            showlegend=False,
        ),
        go.Scatter(
            x=[0],
            y=[0],
            hoverinfo="skip",
            mode="markers",
            marker={"size": 4, "cmax": 1, "cmin": 0, "color": "black"},
            showlegend=False,
        ),
    ]
    layout = dict(
        xaxis={
            "range": [-4, 4],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "showticklabels": False,
        },
        yaxis={
            "range": [-4, 4],
            "showgrid": False,
            "zeroline": False,
            "showline": False,
            "ticks": "",
            "showticklabels": False,
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        width=121,
        height=121,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
    )
    fig = go.Figure(data=data, layout=layout)
    # fig.write_image("/p/project/solai/yang21/hackathon2024/scripts/resource/tem_image.png")
    # Convert the figure to an image (PNG) using Kaleido
    image_bytes = fig.to_image(format="png")

    # Read the image bytes into a NumPy array using Pillow
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("L")
    image_array = np.array(image)

    # Display the shape of the NumPy array
    print("Image shape:", image_array.shape)  # (height, width, 4) where 4 is RGBA channels

    # If needed, display the image using Pillow
    image.save("/p/project/solai/yang21/hackathon2024/scripts/resource/tem_image_io.png")

if __name__ == "__main__":
    generate_diffraction_from_sturcture()
