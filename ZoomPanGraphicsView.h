#pragma once
#ifndef ZOOMPANGRAPHICSVIEW_H
#define ZOOMPANGRAPHICSVIEW_H

#include <QGraphicsView>
#include <QWheelEvent>
#include <QMouseEvent>

class ZoomPanGraphicsView : public QGraphicsView {

    Q_OBJECT // This macro must be present

public:
    ZoomPanGraphicsView(QWidget* parent = nullptr);

protected:
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QPoint lastPanPoint;
    QGraphicsScene* customScene;  // Renamed to avoid conflict
signals:
    void saveRequested(QGraphicsScene* scene);
    void polygonClicked(int polygonId);  // Signal to emit when a polygon is clicked
};

#endif // ZOOMPANGRAPHICSVIEW_H
