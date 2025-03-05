import datetime
import pandas as pd
from evidently.metrics import (
    ColumnDriftMetric, ColumnSummaryMetric, DatasetDriftMetric,
    DatasetMissingValuesMetric, ClassificationQualityMetric
)
from evidently.report import Report
from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import (
    DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType, ReportFilter, CounterAgg
)

# Définition des fichiers de données
REF_DATA_PATH = "../data/ref_data.csv"
PROD_DATA_PATH = "../data/prod_data.csv"
WORKSPACE = "workspace"
PROJECT_NAME = "Model Performance Monitoring"
PROJECT_DESCRIPTION = "Monitoring des performances du modèle avec Evidently."

def load_data():
    ref_data = pd.read_csv(REF_DATA_PATH)
    prod_data = pd.read_csv(PROD_DATA_PATH)

    # Harmoniser les colonnes entre ref_data et prod_data
    ref_data = ref_data.rename(columns={"stroke": "target"})  # Renommer la cible
    ref_data["prediction"] = -1  # Ajouter une colonne "prediction" par défaut

    # Sauvegarder la version mise à jour
    ref_data.to_csv(REF_DATA_PATH, index=False)
    return ref_data, prod_data

def create_report(reference_data, current_data):
    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="age", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="age"),
            ColumnDriftMetric(column_name="bmi", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="bmi"),
            ClassificationQualityMetric() 
        ],
        timestamp=datetime.datetime.now()
    )

    report.run(reference_data=reference_data, current_data=current_data)
    return report

def setup_dashboard(workspace):
    project = workspace.create_project(PROJECT_NAME)
    project.description = PROJECT_DESCRIPTION

    # Nombre total de lignes analysées
    project.dashboard.add_panel(DashboardPanelCounter(
        title="Nombre de lignes analysées",
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        value=PanelValue(metric_id="DatasetMissingValuesMetric", field_path="current.number_of_rows"),
        text="count",
        agg=CounterAgg.SUM
    ))
    
    # Visualisation du Drift des données
    project.dashboard.add_panel(DashboardPanelPlot(
        title="Proportion de variables driftées",
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        values=[
            PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns", legend="Drift Share"),
        ],
        plot_type=PlotType.LINE
    ))

    # Visualisation des performances du modèle
    project.dashboard.add_panel(DashboardPanelPlot(
        title="Performance du modèle (Classification)",
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        values=[
            PanelValue(metric_id="ClassificationQualityMetric", field_path="f1", legend="F1-score"),
            PanelValue(metric_id="ClassificationQualityMetric", field_path="balanced_accuracy", legend="Balanced Accuracy"),
            PanelValue(metric_id="ClassificationQualityMetric", field_path="precision", legend="Precision"),
            PanelValue(metric_id="ClassificationQualityMetric", field_path="recall", legend="Recall"),
        ],
        plot_type=PlotType.LINE
    ))
    
    project.save()
    return project

def main():
    workspace = Workspace.create(WORKSPACE)
    ref_data, prod_data = load_data()
    report = create_report(ref_data, prod_data)
    project = setup_dashboard(workspace)
    workspace.add_report(project.id, report)

if __name__ == "__main__":
    main()