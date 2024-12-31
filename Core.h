#pragma once
#ifndef CORE_H
#define CORE_H

#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>
#include <random>
#include <unordered_map>
#include <queue>
#include <set>
#include <QElapsedTimer>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <QTimer>
#include <unordered_set>
#include "SimpleNode.h"
#include <cmath>
#include <functional>
#include <thread>
#include <mutex>
#include <future>
#include <QImage>
#include <QPixmap>
#include <QObject>
#include <opencv2/opencv.hpp>
#include <utility> // for pair
#include <limits>  // for numeric_limits
#include "gnuplot-iostream.h"

// Define the PairHash structure
struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const;
    size_t operator()(const std::tuple<int, int, int, int>& t) const;
};

struct Node {
    std::vector<double> trait3Values;
    int id;
    double trait3;
    int row;
    int col;
    bool analysed;
    bool isPath;
    bool calculado = false;
    int accessPointID;
};

// Define ranges of values and corresponding colors
struct ColorRange {
    double minValue;
    double maxValue;
    QColor color;
};

struct AgentNode {
    int id;
    double trait2;
    int row;
    int col;
    bool analysed;
    bool isPath;
    int accessPointID;
};

// Define a struct to hold the grid node information
struct GridNodeHeatPath {
    int row;
    int col;
    int score;
    QColor color;
    QPointF point;
};

// Define a struct to hold the grid
struct GridHeatPath {
    std::vector<std::vector<GridNodeHeatPath>> heatnodes;
};

struct Agent {
    std::vector<double> trait2Values;
    double trait2;
    double latentTrait;
    double alpha;
    double beta;
    int row;
    int col;
    int initrow;
    int initcol;
    int currentAccessPointID;
    std::string name;
    std::vector<int> AccessPointIDValues;
};

struct PolygonData {
    int id;
    std::vector<QPointF> points;
    QColor color; // Add color attribute
    std::vector<QColor> labelValues;
    std::vector<double> trait3Values;

    // Conversion operator to convert PolygonData to QPolygonF
    operator QPolygonF() const {
        QPolygonF polygon;
        for (const auto& point : points) {
            polygon << point;
        }
        return polygon;
    }
};

struct PathNode {
    int row, col;
    double costSoFar, heuristic;
    int parentRow, parentCol; // Parent information

    // Default constructor
    PathNode() : row(-1), col(-1), costSoFar(0.0), heuristic(0.0), parentRow(-1), parentCol(-1) {}

    PathNode(int r, int c, double cost, double h, int pr = -1, int pc = -1)
        : row(r), col(c), costSoFar(cost), heuristic(h), parentRow(pr), parentCol(pc) {}

    bool operator<(const PathNode& other) const {
        // Note: In priority queues in C++, the element with the highest priority is given the highest value.
        // Since we want a min-heap (lower costs to have higher priority), we invert the comparison here.
        return costSoFar + heuristic > other.costSoFar + other.heuristic;
    }
};

struct PathInfo {
    double cost;
    std::vector<std::pair<short, short>> path;
    int Length;
    int totalVisitedNodes;
    std::pair<int, int> source;
    std::pair<int, int> destination;

    // Assignment operator for deep copy
    PathInfo& operator=(const PathInfo& other);
};

// Define the ComparePathNode structure
struct ComparePathNode {
    bool operator()(const PathNode& left, const PathNode& right) const;
};

class ABM : public QObject {
    Q_OBJECT

public:
    using pathMap = std::unordered_map<std::tuple<int, int, int, int>, PathInfo, PairHash>;

    std::priority_queue<PathNode, std::vector<PathNode>, ComparePathNode> pq;
    std::map<std::pair<int, int>, PathNode> nodesMap;

    GridHeatPath convertPolygonsToGrid(const std::vector<PolygonData>& polygons, int gridWidth, int gridHeight);
    std::string extractFilename(const std::string fullPath);
    void calculateShortestPathsCUDA(const std::vector<std::vector<Node>>& nodeGrid, int startRow, int startCol, int destRow, int destCol);
    double calculateAverageTrait2Value(const Agent& agent);
    int getHighestVisitedNodes(const pathMap& pathMap);
    double calculateMedianVisitedNodes(const pathMap& pathMap);
    std::pair<double, double> findTraitBounds(const std::vector<std::vector<Node>>& nodeGrid);
    double generateRandomValue(double lowerBound, double upperBound);
    double calculateAgentShannonEntropy(const std::vector<Agent>& agents);
    double calculateShannonEntropyGRID(const std::vector<std::vector<Node>>& nodeGrid);
    double perpendicularDistance(const std::pair<int, int>& point, const std::pair<int, int>& lineStart, const std::pair<int, int>& lineEnd);
    void simplifyPath(std::vector<std::pair<int, int>>& path, double tolerance);
    void savePathsToCSV(const std::string& filename, const pathMap& paths);
    void exportImageFromNodeGrid(const std::vector<std::vector<Node>>& nodeGrid);
    void createImageFromNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, const std::string& outputFilename);
    std::vector<std::string> pegarFields(const char* shapefilePath);
    std::vector<Agent> initializeAgents(const std::vector<std::vector<Node>>& nodeGrid, int numAgents, double minTrait, double maxTrait);
    double calculateScore(const Agent& agent, int row, int col, const Node& targetNode, const pathMap& pathMap, int currentRow, int currentCol, double alpha, double beta, double Maiorcaminho, double maxTrait);
    std::tuple<int, int, int> calculateNextAccessPoint(const Agent& agent, const std::vector<std::vector<Node>>& nodeGrid, const pathMap& pathMap, double alpha, double beta, double Maiorcaminho, double maxTrait, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alphaSteps, int betaSteps);
    void moveAgents(std::vector<Agent>& agents, const std::vector<std::vector<Node>>& nodeGrid,const pathMap& pathMap, double alpha, double beta, double Maiorcaminho, double maxTrait, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alphaSteps, int betaSteps);
    void updateAgentTraits(std::vector<Agent>& agents, std::vector<std::vector<Node>>& nodeGrid, double pesoAgente, double pesoLugar, double minTrait, double maxTrait, std::vector<double>& GRIDentropyValues, std::vector<double>& AGENTSentropyValues);
    void savePaths(const pathMap& paths, const std::string& filename);
    pathMap loadPaths(const std::string& filename);
    void precomputeHeuristics(const std::vector<std::vector<Node>>& nodeGrid, int destRow, int destCol);
    void addNeighborsToQueueAStar(const std::vector<std::vector<Node>>& nodeGrid, std::priority_queue<PathNode, std::vector<PathNode>, ComparePathNode>& pq, std::map<std::pair<int, int>, PathNode>& nodesMap, std::vector<bool>& visited, int row, int col, double costSoFar, int parentRow, int parentCol, int destRow, int destCol);
    void calculateShortestPathsAStar(const std::vector<std::vector<Node>>& nodeGrid, int startRow, int startCol, int destRow, int destCol, pathMap& pathMap);
    std::vector<std::vector<PathInfo>> calculateShortestPaths(const std::vector<std::vector<Node>>& nodeGrid, pathMap& pathMap);
    void saveNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, const std::string& filename, int numRows, int numCols);
    std::vector<std::vector<Node>> loadNodeGrid(const std::string& filename);
    void processMap(std::vector<std::vector<Node>>& nodeGrid, int numRows, int numCols, pathMap& pathMap, std::string filenameToSave, int option, int numAgents, int interactions, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, std::vector<PolygonData> poligonos);
    void printNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, int numRows, int numCols);
    void processShapefile(const char* shapefilePath, const char* shapefilePath2, const std::string filenameToSave, std::string CAMINHODAPASTA, double nodeSize, std::string NOMEDOUSO, int option, int numAgents, int interactions, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, bool node, int VoltarPraCasa, int DistribuicaoUsos, int Maxtraits, int MaxtraitsAgents, int DistribuicaoAgentes, int threads, int metodoEscolha, std::unordered_map<int, int> idTraitMap, std::vector<std::pair<int, int>> agentTraits);
    void chooseRunMethod(std::vector<std::vector<Node>>& nodeGrid, pathMap& pathMap, std::string filenameToSave, int option, int numAgents, int interactions, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, std::vector<PolygonData> poligonos);
    int batelada(std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, pathMap& pathMap, double minTrait, double maxTrait, std::string filenameToSave, int numAgents, int numIterations, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, std::vector<std::vector<std::vector<Agent>>>& AgentsVector);
    void plotHeatmapGNU(std::ofstream& BateladaPlotScript, const std::vector<std::vector<double>>& heatmapData, const std::string& title, const std::vector<double>& xLabels, const std::vector<double>& yLabels, const std::string& filenameToSave, int numAgents, int numIterations, int steps, int stepSize, int alpha, int beta, int pesoAgente, int pesoLugar);
    std::vector<std::vector<std::tuple<double, double>>> fillEntropyValues( int alphaSteps, int betaSteps, int numIterations, std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, const pathMap& pathMap, double stepSize, double alpha, double beta, double pesoAgente, double pesoLugar, double Maiorcaminho, std::string filenameToSave, double maxTrait, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, std::vector<std::vector<std::vector<Agent>>>& AgentsVector);
    std::tuple <std::tuple<double, double>, bool, std::vector<double>, std::vector<double>, int, std::vector<double>, double, double, std::vector<std::vector<double>>, std::vector<std::vector<double>>> calculateEntropyAgentsBatelada( double alpha, double beta, int numIterations, std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents,const pathMap& pathMap, double stepSize, double pesoAgente, double pesoLugar, double Maiorcaminho, std::string filenameToSave, int contador, std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVector, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector,int alphaSteps, int betaSteps);
    void run(std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, pathMap& pathMap, double minTrait, double maxTrait, std::string filenameToSave, int numAgents, int numIterations, double alpha, double beta, double pesoAgente, double pesoLugar, std::vector<PolygonData> poligonos, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector);
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> plotSingleGraphsQT(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait);
    void plotSingleGraphsGNU(std::ofstream& BateladaPlotScript, bool Batelada , std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait);
    void plotSingleGraphsGNU2(std::ofstream& BateladaPlotScript, bool Batelada , std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait);
    void plotSingleGraphsGNU3(std::ofstream& BateladaPlotScript, bool Batelada , std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait, std::vector<std::vector<double>> agentTraitValues);
    std::vector<std::pair<int, int>> reconstructShortestPath(pathMap& pathMap, int startRow, int startCol, int destRow, int destCol);
    void plotSingleGraphsGNUCMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait);
    void plotSingleGraphsGNU2CMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait);
    void plotSingleGraphsGNU3CMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait, std::vector<std::vector<double>> agentTraitValues);
    void generatePathImage(const std::vector<std::vector<Node>>& nodeGrid, const std::vector<std::pair<int, int>>& path, std::string title);
    void inicio(std::string shapefilePath, std::string CAMINHOPASTA, int recSize, int option, int numAgents, int interactions, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, int VoltarPraCasa, int DistribuicaoUsos, int Maxtraits, int MaxtraitsAgents, int DistribuicaoAgentes, int threads, int metodoEscolha, std::unordered_map<int, int> idTraitMap, std::vector<std::pair<int, int>> agentTraits);
    std::vector<PolygonData> criarPoligonos(const char* shapefilePath, std::string NOMEDOUSO);
    QColor getColorForValue(double labelValue, double highestLabelValue);
    void savePolygonsToFile(const std::vector<PolygonData>& polygons, const std::string& filename);
    std::vector<PolygonData> loadPolygonsFromFile(const std::string& filename);
    void updateGridHeatPath(int sourceRow, int sourceCol, int destinationRow, int destinationCol, const pathMap& pathmap, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alpha, int beta);
    void calculateNodeColors(GridHeatPath& gridHeatPath);
    int calculateUniquePairs(int n);
    void carregarPoligonos(const std::string& shapefilePath);

signals:
    void ImageProduced(const QString& fileName);
    void ImageProduced2(const QString& fileName2, const QString& fileName3, const QString& fileName4, const QString& fileName5, const QString& fileName6);
    void ImageProduced3(const QString& fileName);
    void progressUpdated(float progress);
    void progressUpdated2(float progress, QString estimatedTimeRemaining);
    void logMessage(const QString& message);
    void graphDataReady(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::vector<double> GRIDentropyValues, std::vector<std::vector<double>> agentTraitValues, double minTrait, double maxTrait, std::vector<std::vector<double>>  nodesTraitValues);
    void graphHeatMap(const std::vector<std::vector<double>> firstValues, std::vector<std::vector<double>> secondValues );
    void messageBoxBegin();
    void messageBoxEnd();
    void salvarplotsinterno(std::vector<std::vector<bool>> Batelada, std::vector<std::vector<std::vector<double>>> iterationNumbers, std::vector<std::vector<std::vector<double>>> AGENTSentropyValues, std::vector<std::vector<int>> numAgents, std::vector<std::vector<std::vector<double>>> GRIDentropyValues, std::vector<std::vector<double>> minTrait, std::vector<std::vector<double>> maxTrait, std::vector<std::vector<std::vector<std::vector<double>>>> agentTraitValuesContainer, std::vector<std::vector<std::vector<std::vector<double>>>> nodesTraitValuesContainer);
    void polygonsReady(std::vector<PolygonData> polygons);
    void colorsForTraitValuesEmitted(std::vector<std::pair<int, std::vector<QColor>>> idColorsPairs, std::vector<std::pair<int, std::vector<double>>> traitsPoligono);
    void colorsForTraitValuesEmittedBatelada(std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVector, std::vector<std::vector<std::vector<std::pair<int, std::vector<double>>>>> traitsPoligonoVector);
    void emitirtiposDeUso(double maxTrait, std::vector<QColor> colorVector);
    void agentesIniciados(std::vector<Agent> agents);
    void exportarHeatPath(GridHeatPath gridHeatPath);
    void mandargridHeatPathVector(std::vector<std::vector<GridHeatPath>> gridHeatPathVector);
    void agentesIniciadosVector(std::vector<std::vector<std::vector<Agent>>> AgentsVector);
    void graphHeatMapQT(const std::vector<std::vector<double>>& heatmapData, const QString& title, const std::vector<double>& xLabels, const std::vector<double>& yLabels);
    void criarMosaico();
private:
    QElapsedTimer timer; // For estimating time to complete
    std::mutex mergeMutex;
    int totalTasks;
    int VoltarPraCasaLocal = 0;
    int DistribuicaoUsosLocal = 0;
    int MaxtraitsLocal = 0;
    int MaxtraitsAgentsLocal = 0;
    int DistribuicaoAgentesLocal = 0;
    int threadsLocal = 2;
    int metodoEscolhaLocal;
    std::vector<std::pair<int, int>> agentTraitsLocal;
    std::unordered_map<int, int> idTraitMapLocal;
    // Create the unordered_map to store accessPointID as the key, and a pair of (row, col) as the value
    std::unordered_map<int, std::pair<int, int>> accessPointMap;
    //void calculateShortestPathsAStar(const std::vector<std::vector<Node>>& nodeGrid, int startRow, int startCol, int destRow, int destCol, pathMap& pathMap);
};

#endif // CORE_H