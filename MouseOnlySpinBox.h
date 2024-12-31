#pragma once
#include <QSpinBox>
#include <QKeyEvent>  // Include this line to resolve the incomplete type error

class MouseOnlySpinBox : public QSpinBox {
    Q_OBJECT

public:
    MouseOnlySpinBox(QWidget* parent = nullptr) : QSpinBox(parent) {}

protected:
    void keyPressEvent(QKeyEvent* event) override {
        // Ignore all key press events
        event->ignore();
    }
};
