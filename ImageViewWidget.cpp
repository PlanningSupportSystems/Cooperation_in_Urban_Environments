// ImageViewWidget.cpp

#include "ImageViewWidget.h"
#include <qscrollarea>
#include <qscrollbar>

ImageViewWidget::ImageViewWidget(QWidget* parent) : QWidget(parent) {
    // Initialize imageLabel and add it to the layout of this widget
    imageLabel = new QLabel(this);
    layout = new QVBoxLayout(this);
    layout->addWidget(imageLabel);

    // Initialize variables for zoom and pan
    isPanning = false;
    isZooming = false;
}

QVBoxLayout* ImageViewWidget::getLayout() const {
    return layout;
}

// Implement mouse event functions...


void ImageViewWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        lastPanPosition = event->pos();
        isPanning = true;
    }
    else if (event->button() == Qt::RightButton) {
        lastZoomPosition = event->pos();
        isZooming = true;
    }
}

void ImageViewWidget::mouseMoveEvent(QMouseEvent* event) {
    if (isPanning) {
        QPoint delta = event->pos() - lastPanPosition;
        QScrollArea* scrollArea = qobject_cast<QScrollArea*>(imageLabel->parentWidget());
        if (scrollArea) {
            scrollArea->horizontalScrollBar()->setValue(scrollArea->horizontalScrollBar()->value() - delta.x());
            scrollArea->verticalScrollBar()->setValue(scrollArea->verticalScrollBar()->value() - delta.y());
        }
        lastPanPosition = event->pos();
    }
    else if (isZooming) {
        QPoint delta = event->pos() - lastZoomPosition;
        qreal scaleFactor = qPow(1.01, delta.y() / 120.0); // Adjust the zoom sensitivity
        imageLabel->resize(imageLabel->size() * scaleFactor);
        lastZoomPosition = event->pos();
    }
}


void ImageViewWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isPanning = false;
    }
    else if (event->button() == Qt::RightButton) {
        isZooming = false;
    }
}
