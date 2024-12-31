#pragma once
#include <QProgressBar>
#include <QDialog>
#include <QPushButton>
#include <QVBoxLayout>

class ProgressDialog : public QDialog {
    Q_OBJECT  // Add this macro to enable Qt meta-object features

public:
    ProgressDialog(QWidget* parent = nullptr)
        : QDialog(parent), progressBar(new QProgressBar(this)), okButton(new QPushButton("OK", this)) {

        okButton->setEnabled(false);  // Disable until the task is complete

        QVBoxLayout* layout = new QVBoxLayout(this);
        layout->addWidget(progressBar);
        layout->addWidget(okButton);

        connect(okButton, &QPushButton::clicked, this, &ProgressDialog::accept);
        setLayout(layout);
    }

    QProgressBar* getProgressBar() { return progressBar; }
    QPushButton* getOkButton() { return okButton; }

private:
    QProgressBar* progressBar;
    QPushButton* okButton;
};
