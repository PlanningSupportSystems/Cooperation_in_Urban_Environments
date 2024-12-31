#include <QWheelEvent>
#include <QMouseEvent>
#include <QApplication>
#include <QPushButton>
#include <QGraphicsPolygonItem>

#include "ZoomPanGraphicsView.h"

ZoomPanGraphicsView::ZoomPanGraphicsView(QWidget* parent) : QGraphicsView(parent) {
    customScene = new QGraphicsScene(this);  // Create the scene
    this->setScene(customScene);  // Set the scene for this view

    setDragMode(QGraphicsView::ScrollHandDrag);
    setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
    setResizeAnchor(QGraphicsView::AnchorUnderMouse);
    setRenderHint(QPainter::Antialiasing, true);

    QPushButton* saveSvgButton = new QPushButton("Save as SVG", this);
    saveSvgButton->move(10, 10);

    connect(saveSvgButton, &QPushButton::clicked, this, [this]() {
        QGraphicsScene* currentScene = this->scene();  // Use QGraphicsView's scene() method
        if (currentScene) {
            emit saveRequested(currentScene);  // Emit the signal with the scene pointer
        }
        });
}

void ZoomPanGraphicsView::wheelEvent(QWheelEvent* event) {
    if (event->modifiers() == Qt::ControlModifier) { // Zoom with Ctrl key pressed
        double scaleFactor = 1.15;
        if (event->angleDelta().y() < 0) {
            scaleFactor = 1.0 / scaleFactor;
        }
        scale(scaleFactor, scaleFactor);
    }
    else {
        QGraphicsView::wheelEvent(event); // Let the base class handle other wheel events
    }
}

void ZoomPanGraphicsView::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton) { // Start panning with middle mouse button
        lastPanPoint = event->pos();
    }
    else {
        QGraphicsView::mousePressEvent(event); // Let the base class handle other mouse press events
        // Check if the event is a left-click
        if (event->button() == Qt::RightButton) {
            // Map the click position to the scene
            QPointF scenePos = mapToScene(event->pos());

            // Get the item at the clicked position
            QGraphicsItem* item = scene()->itemAt(scenePos, transform());

            // Check if the item is a polygon
            if (QGraphicsPolygonItem* polygonItem = dynamic_cast<QGraphicsPolygonItem*>(item)) {
                // Get the polygon ID (assuming it's stored in the data(0))
                int polygonId = polygonItem->data(0).toInt();

                // Emit the polygonClicked signal
                emit polygonClicked(polygonId);
            }
        }
    }
}

void ZoomPanGraphicsView::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::MiddleButton) { // Pan if middle mouse button is held down
        QPointF delta = mapToScene(event->pos()) - mapToScene(lastPanPoint);
        translate(delta.x(), delta.y());
        lastPanPoint = event->pos();
    }
    else {
        QGraphicsView::mouseMoveEvent(event); // Let the base class handle other mouse move events
    }
}

void ZoomPanGraphicsView::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::MiddleButton) { // Stop panning when middle mouse button is released
        QApplication::restoreOverrideCursor();
    }
    else {
        QGraphicsView::mouseReleaseEvent(event); // Let the base class handle other mouse release events
    }
}
