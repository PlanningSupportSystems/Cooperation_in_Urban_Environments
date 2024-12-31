#pragma once
#ifndef CUSTOMWIDGET_H
#define CUSTOMWIDGET_H

#include <QWidget>
#include <QChartView>
#include <QPushButton>
#include <QVBoxLayout>
#include <QChart>
#include <QtWidgets/QFileDialog>
#include "URBAN_ABM_QT.h"

//QT_CHARTS_USE_NAMESPACE

class CustomWidget : public QWidget {
    Q_OBJECT
public:
    explicit CustomWidget(QChart* chart, QWidget* parent = nullptr) : QWidget(parent) {
        auto* chartView = new QChartView(chart);
        auto* button = new QPushButton("Save as SVG");

        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->addWidget(chartView);
        layout->addWidget(button);

        // Connect the button's click signal to the slot that saves the chart
        connect(button, &QPushButton::clicked, this, [chart, this]() {
            emit saveRequested(chart, QSize(1920, 1080)); // Emit the signal
            
        });
    }

signals:
    void saveRequested(QChart* chart, const QSize& size);
};

#endif // CUSTOMWIDGET_H