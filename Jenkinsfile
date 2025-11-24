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

        stage('Validate Python Syntax for newdas DAGs') {
            steps {
                echo "Validating DAG Python files in newdas/dags..."
                sh '''
                if command -v python3 >/dev/null 2>&1; then
                    python3 -m py_compile newdas/dags/*.py
                else
                    echo "Python3 not found → skipping syntax validation"
                fi
                '''
            }
        }

        stage('Deploy newdas DAGs to Airflow') {
            steps {
                echo "Deploying DAGs from newdas/dags to Airflow DAG folder..."
                sh '''
                mkdir -p ${AIRFLOW_DAG_FOLDER}

                rm -f ${AIRFLOW_DAG_FOLDER}/*.py

                cp newdas/dags/*.py ${AIRFLOW_DAG_FOLDER}/

                echo "Deployment completed successfully!"
                '''
            }
        }
    }

    post {
        success {
            echo "✅ DAG CI/CD Pipeline SUCCESS — Airflow will refresh in ~30 seconds."
        }
        failure {
            echo "❌ DAG CI/CD Pipeline FAILED — Check Jenkins console logs."
        }
    }
}

