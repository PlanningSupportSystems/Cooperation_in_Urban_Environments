#pragma once
#include <QWidget>
#include <QChartView>
#include <QPushButton>
#include <QVBoxLayout>
#include <QChart>
#include <QtWidgets/QFileDialog>
#include <QResizeEvent>
#include "URBAN_ABM_QT.h"

class HeatmapWidget : public QWidget {
    Q_OBJECT

private:
    QGraphicsView* view; // Store the view as a member variable to access it in resizeEvent

public:
    HeatmapWidget(QGraphicsScene* scene, QWidget* parent = nullptr) : QWidget(parent), view(new ZoomPanGraphicsView(this)) {
        auto* layout = new QVBoxLayout(this);
        layout->addWidget(view);
        view->setScene(scene);  // Set the scene for the custom view
        auto* button = new QPushButton("Save as SVG");
        layout->addWidget(button);

        // Connect the button's click signal to the saveRequested signal
        connect(button, &QPushButton::clicked, this, [scene, this]() {
            emit saveRequested2(scene);  // Emit the signal when the button is clicked
            });
    }

    void setupScene(const std::vector<std::vector<double>>& data, const std::vector<double>& xLabels, const std::vector<double>& yLabels) {
        void createHeatmapWithRainbowColormap(const std::vector<std::vector<double>>&heatmapData, const QString & title, const std::vector<double>&xLabels, const std::vector<double>&yLabels, const QString & filenameToSave);
        //adjustViewToFitScene();
    }
    void adjustViewToFitScene() {
        if (view && view->scene()) {
            // Reset the view scale to 1:1
            view->resetTransform();
            // Fit the scene's content within the view while maintaining aspect ratio
            view->fitInView(view->scene()->sceneRect(), Qt::KeepAspectRatio);
            //view->scale(45, 45); // Adjust the scaling factor as needed
        }
    }
protected:
    void resizeEvent(QResizeEvent* event) override {
        QWidget::resizeEvent(event);
        // Adjust the view to fit the scene content after resizing
        adjustViewToFitScene();
    }
    void showEvent(QShowEvent* event) override {
        QWidget::showEvent(event);
        // Adjust the view to fit the scene content after the widget is shown
        adjustViewToFitScene();
    }
signals:
    void saveRequested2(QGraphicsScene* scene);
};