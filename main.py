import click
import subprocess

@click.group()
def cli():
    pass

@click.command()
@click.option('--extract-hog', type=bool, default=True, help="Generate HOG features")
@click.option('--extract-canny-edges', type=bool, default=True, help="Generate Canny Edges features")
@click.option('--extract-contours', type=bool, default=True, help="Generate Contours features")
@click.option('--extract-vgg19', type=bool, default=True, help="Generate VGG19 features")
@click.option('--extract-pca', type=bool, default=True, help="Generate PCA features")
def run_data_pipeline(extract_hog, extract_canny_edges, extract_contours, extract_vgg19, extract_pca):
    scripts = ["load_data.py", "preprocessing.py"]
    if extract_hog:
        scripts.append("extract_hog.py")
    if extract_canny_edges:
        scripts.append("extract_canny_edges.py")
    if extract_contours:
        scripts.append("extract_contours.py")
    if extract_vgg19:
        scripts.append("extract_vgg19.py")
    if extract_pca:
        scripts.append("extract_pca.py")

    for script in scripts:
        script_path = f"scripts/{script}"
        result = subprocess.run(["python", script_path], capture_output=False, text=False)
        if result.returncode != 0:
            print(f"Error running {script_path}: {result.stderr}")
            break

cli.add_command(run_data_pipeline)

if __name__ == "__main__":
    cli()
