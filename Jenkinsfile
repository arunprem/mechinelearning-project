pipeline {
    agent any

    environment {
        AIRFLOW_DAG_FOLDER = "/airflow-dags"
    }

    stages {

        stage('Checkout Code') {
            steps {
                echo "Pulling code from development branch..."
                git branch: 'development',
                    url: 'https://github.com/arunprem/mechinelearning-project.git'
            }
        }

        stage('Validate Python Syntax') {
            steps {
                echo "Validating DAG Python files..."
                sh '''
                if command -v python3 >/dev/null 2>&1; then
                    python3 -m py_compile newdags/dags/*.py
                else
                    echo "Python3 not available → skipping syntax validation"
                fi
                '''
            }
        }

        stage('Deploy DAGs to Airflow') {
            steps {
                echo "Deploying DAGs into Airflow DAG folder..."
                sh '''
                mkdir -p ${AIRFLOW_DAG_FOLDER}

                # Remove old DAGs
                rm -f ${AIRFLOW_DAG_FOLDER}/*.py

                # Copy new DAGs from newdags/dags
                cp newdags/dags/*.py ${AIRFLOW_DAG_FOLDER}/

                echo "🚀 DAG deployment completed successfully!"
                '''
            }
        }
    }

    post {
        success {
            echo "✅ DAG CI/CD SUCCESS — Airflow will refresh soon."
        }
        failure {
            echo "❌ DAG CI/CD FAILED — Check Jenkins logs."
        }
    }
}
