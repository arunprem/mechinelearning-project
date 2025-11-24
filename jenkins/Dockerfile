FROM jenkins/jenkins:lts

USER root

# Install Python packages (Debian Trixie compatible)
RUN apt-get update && apt-get install -y \
      python3 \
      python3-pip \
      python3-venv \
      && rm -rf /var/lib/apt/lists/*

# Make python command available
RUN ln -s /usr/bin/python3 /usr/bin/python || true

# Install MLflow and dependencies using pip override for PEP 668
RUN pip3 install --break-system-packages mlflow boto3 pandas numpy requests

# ---------------------------------------------------
# FIX: Jenkins needs the docker group to exist first
# ---------------------------------------------------
RUN groupadd docker
RUN usermod -aG docker jenkins

# Fix macOS volume permission issues
RUN groupadd -g 1000 hostgroup || true
RUN usermod -aG 1000 jenkins

USER jenkins
