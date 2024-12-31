#pragma once
#ifndef URBAN_ABM_QT_H
#define URBAN_ABM_QT_H

#include <QtWidgets/QMainWindow>
#include "ui_URBAN_ABM_QT.h"
#include "Core.h"
#include <QTextEdit>
#include <Qchart>
#include "ZoomPanGraphicsView.h"
#include <QChartView>



class URBAN_ABM_QT : public QMainWindow
{
    Q_OBJECT
    std::vector<PolygonData> polygonsLocal;
    std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVectorLocal;
    std::vector<std::vector<std::vector<std::pair<int, std::vector<double>>>>> traitsPoligonoVectorLocal;
    std::vector<std::pair<int, std::vector<QColor>>> idColorsPairsLocal;
    std::vector<std::pair<int, std::vector<double>>> traitsPoligonoLocal;
    GridHeatPath gridHeatPathLocal;
    ZoomPanGraphicsView* zoomPanView;

public slots:
    void openOrientationWindow(int polygonId);
    void updateProgressBar(float progress);
    void updateProgressBarTimeEstimation(float progress, QString estimatedTimeRemaining);
    void updateImage(const QString& fileName);
    void updateHEATMAPIMAGE(const QString& fileName);
    void updateImage3(const QString& fileName, const char* titulo, int numero, const char* tipo);
    void updateImage2(const QString& fileName2, const QString& fileName3, const QString& fileName4, const QString& fileName5, const QString& fileName6);
    void appendToTextEdit(const QString& message);
    void mudancaTipoSimulacao(int index);
    void liberarTabelasdeOrientacoesAgentes(int index);
    void liberarTabelasdeOrientacoesLocais(int index);
    void CarregarVariaveisLocais();
    void updateGraph(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::vector<double> GRIDentropyValues, std::vector<std::vector<double>> agentTraitValues, double minTrait, double maxTrait, std::vector<std::vector<double>>  nodesTraitValues);
    void plotHeatmapQT(const std::vector<std::vector<double>> firstValues, std::vector<std::vector<double>> secondValues);
    void plotHeatmapQT2(const std::vector<std::vector<double>> Values, const char* titulo);
    void createAgentCircles();
    QPointF calculatePolygonCentroid(const PolygonData& polygon);
    bool pointInsidePolygon(const QPointF& point, const std::vector<QPointF>& polygonPoints);
    QPointF findNearestPointInsidePolygon(const QPointF& point, const std::vector<QPointF>& polygonPoints);
    void createHeatmapSquares();
    void saveChartToSVG(QChart* chart, const QSize& size = QSize(1920, 1080));
    void saveSceneToSVG(QGraphicsScene* scene);
    void saveMapToSVG(QGraphicsScene* scene);
    QColor valueToRainbowColor(double value, double min, double max);
    void createHeatmapWithRainbowColormap(const std::vector<std::vector<double>>& heatmapData, const QString& title, const std::vector<double>& xLabels, const std::vector<double>& yLabels);
    void saveDataToFile();
    void loadDataFromFile();
    void carregarPoligonos();
    void openPolygonTraitWindow();
    void showAgentsTable();
    void applySchellingModel(int iterations, double toleranceThreshold, double toleranceThreshold2);
    //std::vector<int> getNeighboringPolygons(int polygonId);
    void updatePolygonColors();
    void openSchellingModelWindow();
    void populateIdTraitMap();
    void exportSVGAnimation();
    void saveSceneToSVGInMemory(QGraphicsScene* scene, QString& svgData);

    std::tuple<std::string, double, double> extractInfo(const std::string& filename);
    std::string encodeImageToBase64(const std::string& path);
    void generateCombinedSVG(const std::map<std::string, std::vector<std::tuple<std::string, double, double>>>& pngGroups, const std::string& outputPath);
    void processImagesForSVG(const std::string& directory, const std::string& outputPath);

    void saveAllGraphsAsSVGs();
    void saveChartToSVGSilent(QChart* chart, const QString& filePath, const QSize& size);
    void prepareGraphDataStorage();

    void alertBegin();
    void alertEnd();
    //void renderScene(const std::vector<PolygonData>& polygons);
    //void animateScene(const std::vector<int>& labelValues);
    //void updatePolygonLabelValues(const std::vector<std::pair<int, std::vector<QColor>>>& idColorsPairs, std::vector<PolygonData> polygonsLocalCore);
    //void animateScene();

public:
    URBAN_ABM_QT(QWidget* parent = nullptr);
    ~URBAN_ABM_QT();
    void updatePolygonLabelValues(std::vector<std::pair<int, std::vector<QColor>>> idColorsPairs, std::vector<std::pair<int, std::vector<double>>> traitsPoligono);
    void updatePolygonLabelValuesBatelada(std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVector, std::vector<std::vector<std::vector<std::pair<int, std::vector<double>>>>> traitsPoligonoVector);
    void animateScene();
    QColor getColorForValue(double value, double highestLabelValue);
    void addLabelsAndLegendToScene(QGraphicsScene* scene, const std::vector<double>& xLabels, const std::vector<double>& yLabels,
        const QString& title, double minValue, double maxValue, int cellWidth, int cellHeight);

    std::map<int, std::vector<int>> getNeighboringPolygons(double tolerance = 1.0) {
        std::map<int, std::vector<int>> neighboringPolygonsMap;

        for (size_t i = 0; i < polygonsLocal.size(); ++i) {
            for (size_t j = i + 1; j < polygonsLocal.size(); ++j) {
                if (arePolygonsNeighbors(polygonsLocal[i], polygonsLocal[j], tolerance)) {
                    neighboringPolygonsMap[polygonsLocal[i].id].push_back(polygonsLocal[j].id);
                    neighboringPolygonsMap[polygonsLocal[j].id].push_back(polygonsLocal[i].id);
                }
            }
        }

        return neighboringPolygonsMap;
    }

    void createTraitsGraph(const std::vector<std::vector<std::vector<std::vector<double>>>>& traitValues, const QString& graphTitle, double minTrait, double maxTrait);
    void createCombinedTraitsGraph(double minTrait, double maxTrait);
    void createEntropyHeatmap(const QString& graphType);

    QChartView* createGraphView(std::vector<double> iterationNumbers,
        std::vector<double> AGENTSentropyValues,
        std::vector<double> GRIDentropyValues,
        std::vector<double> agentsTraitValues,
        std::vector<double> nodesTraitValues);

    void createScaledHeatmaps();

private slots:
    void BotaoComecar(); // Slot for pushButton clicked
    void BotaoImportar(); // Slot for import action triggered
    void BotaoLoad(); // Slot for import action triggered
    void updateComboBox2(const std::vector<std::string>& fieldNames);
    void botaoBrowse();
    void botaoPlots();
    void renderScene(std::vector<PolygonData> polygons);
    //void animateScene();
    void toggleAnimation();
    void setarUsos(double maxTrait, std::vector<QColor> colorVector);
    void setColorFromPicker(QPushButton* button, int tipo);
    void salvarOsPlots(const std::vector<std::vector<bool>> Batelada, const std::vector<std::vector<std::vector<double>>> iterationNumbers, const std::vector<std::vector<std::vector<double>>> AGENTSentropyValues, const std::vector<std::vector<int>> numAgents, const std::vector<std::vector<std::vector<double>>> GRIDentropyValues, const std::vector<std::vector<double>> minTrait, const std::vector<std::vector<double>> maxTrait, const std::vector<std::vector<std::vector<std::vector<double>>>> agentTraitValuesContainer, const std::vector<std::vector<std::vector<std::vector<double>>>> nodesTraitValuesContainer);
    void mexeuNoSlider(int value);
    void receberAgentes(std::vector<Agent> agents);
    void receberHeatPaths(GridHeatPath gridHeatPath);
    void receberHeatPathsVector(std::vector<std::vector<GridHeatPath>> gridHeatPathVector);
    void updateAgentCircles();
    void receberAgentesVector(std::vector<std::vector<std::vector<Agent>>> AgentsVector);
    QPointF calculateAdjustedCenter(const PolygonData& polygon, double circleRadius);
    

private:
    Ui::URBAN_ABM_QTClass ui;
    ABM abm;
    QLabel* imageLabel;
    QLabel* imageLabel2;
    QLabel* imageLabel3;
    QLabel* imageLabel4;
    QLabel* imageLabel5;
    QLabel* imageLabel6;
    QLabel* imageLabel7;
    // Declare the QGraphicsView member using ZoomPanGraphicsView
    
    QGraphicsScene* scene;
    QTimer* timer; // Declare QTimer as a member variable
    bool isAnimationPaused;
    int currentFrame;
    static QGraphicsScene* scenePointer;
    //std::vector<PolygonData> polygonsLocal;
    //QTimer* timer;
    //QChart* chartView;
    //std::streambuf* coutBuffer; // Stream buffer for cout redirection
    bool arePolygonsNeighbors(const PolygonData& polygon1, const PolygonData& polygon2, double tolerance) const {
        QPolygonF poly1 = polygon1;
        QPolygonF poly2 = polygon2;

        for (int i = 0; i < poly1.size(); ++i) {
            QPointF pointA1 = poly1[i];
            QPointF pointA2 = poly1[(i + 1) % poly1.size()];

            for (int j = 0; j < poly2.size(); ++j) {
                QPointF pointB1 = poly2[j];
                QPointF pointB2 = poly2[(j + 1) % poly2.size()];

                if (distanceBetweenEdges(pointA1, pointA2, pointB1, pointB2) <= tolerance) {
                    return true;
                }
            }
        }

        return false;
    }
    double distanceBetweenEdges(const QPointF& A1, const QPointF& A2, const QPointF& B1, const QPointF& B2) const {
        QPointF midpointA = (A1 + A2) / 2;
        QPointF midpointB = (B1 + B2) / 2;
        return std::sqrt(std::pow(midpointA.x() - midpointB.x(), 2) + std::pow(midpointA.y() - midpointB.y(), 2));
    }
};

#endif // URBAN_ABM_QT_H