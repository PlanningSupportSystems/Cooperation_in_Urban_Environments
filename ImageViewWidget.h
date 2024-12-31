#pragma once

// ImageViewWidget.h

#ifndef IMAGEVIEWWIDGET_H
#define IMAGEVIEWWIDGET_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QMouseEvent>
#include <qmath.h>

class ImageViewWidget : public QWidget {
    Q_OBJECT
public:
    explicit ImageViewWidget(QWidget* parent = nullptr);

    QVBoxLayout* getLayout() const; // Public function to access the layout

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QLabel* imageLabel;
    QVBoxLayout* layout;
    QPoint lastPanPosition;
    QPoint lastZoomPosition;
    bool isPanning;
    bool isZooming;
};

#endif // IMAGEVIEWWIDGET_H
