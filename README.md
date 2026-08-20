# PEOPLE-ECCO Marine Solutions

Repository that covers PEOPLE-ECCO Marine Solutions, developed by DHI.

It includes the following algorithms:

* *Submerged Aquatic Vegetation and Connectivity*: `src` directory
* *Submerged Aquatic Vegetation and Frequency*: `frequency` directory

## Development

### Environment

Running as CWL locally Workflow requires `docker` and [`cwlref-runner` (Link)](https://github.com/common-workflow-language/cwltool).

*Installing cwltool in Python virtual environment*:

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install cwlref-runner
```

## SAV and Connectivity Workflow

SAV used openEO to fetch / subset matching satellite imagery, post-processing and classification is done via ML prediction model.

### Run with CWL

```bash

# Build Docker Image
docker build -f tooling/sav/sav.Dockerfile \
    -t ghcr.io/people-ecco/dhi-sav:latest .

# set variables
export OPENEO_AUTH_CLIENT_ID="..."
export OPENEO_AUTH_CLIENT_SECRET="..."
export OUTPUT_PATH="./target/cwl_output"

# Run using cwl-runner
export $(cat .env | xargs) && cwl-runner cwl/sav.cwl --cdse_client_id=${OPENEO_AUTH_CLIENT_ID} --cdse_client_secret=${OPENEO_AUTH_CLIENT_SECRET} --parameters tooling/sav/sav_run_parameters_malaysia.json --run_name sav_run
```

### Run Without CWL

```bash
export OPENEO_AUTH_CLIENT_ID="..."
export OPENEO_AUTH_CLIENT_SECRET="..."

python tooling/sav/run_sav.py \
    --parameters tooling/sav/sav_run_parameters_malaysia.json \
    --output-dir ./target/output \
    --run-name sav_run_01
```

