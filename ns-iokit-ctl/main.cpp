#include <iostream>

#include "openpnp-capture/common/logging.h"
#include "openpnp-capture/include/openpnp-capture.h"
#include "openpnp-capture/mac/uvcctrl.h"

int main(int argc, char *argv[]) {
    // Support both legacy 6-arg form and extended 9-arg form
    // argc includes program name: 7 == 6 params, 10 == 9 params
    if (argc != 7 && argc != 11) {
        std::cerr << "Usage (short): ns_iokit_ctl <vid> <pid> <location> <disable autoexposure> <exposure> <gain>" << std::endl;
        std::cerr << "Usage (full):  ns_iokit_ctl <vid> <pid> <location> <disable autoexposure> <exposure> <gain> <saturation> <hue> <whitebalance>" << std::endl;
        std::cerr << "\tvid: IOKit Vendor ID in hex" << std::endl;
        std::cerr << "\tpid: IOKit Product ID in hex" << std::endl;
        std::cerr << "\tlocation: IOKit Location ID in hex, or 0 to select the first found location" << std::endl;
        std::cerr << "\tdisable autoexposure: 1 to disable auto exposure, 0 to enable" << std::endl;
        std::cerr << "\texposure: exposure value, decimal" << std::endl;
        std::cerr << "\tgain: gain value, decimal" << std::endl;
        std::cerr << "\tsaturation: saturation value, 0-100 (optional)" << std::endl;
        std::cerr << "\thue: hue value, 0-100 (optional)" << std::endl;
        std::cerr << "\tdisable autowhitebalance: 1 to disable auto white balance, 0 to enable (optional)" << std::endl;
        std::cerr << "\twhitebalance: white balance value, 0-100 (optional)" << std::endl;
        return 1;
    }

    unsigned int vid = std::stoul(argv[1], nullptr, 16);
    unsigned int pid = std::stoul(argv[2], nullptr, 16);
    unsigned int location = std::stoul(argv[3], nullptr, 16);
    int disableAutoexposure = std::stoi(argv[4], nullptr, 10);
    int exposure = std::stoi(argv[5], nullptr, 10);
    int gain = std::stoi(argv[6], nullptr, 10);

    int saturation = -1;
    int hue = -1;
    int whiteBalance = -1;
    bool disableAutoWhiteBalance = false;
    if (argc == 11) {
        saturation = std::stoi(argv[7], nullptr, 10);
        hue = std::stoi(argv[8], nullptr, 10);
        disableAutoWhiteBalance = std::stoi(argv[9], nullptr, 10) == 0;
        whiteBalance = std::stoi(argv[10], nullptr, 10);
    }

    setLogLevel(LOG_DEBUG);
    std::shared_ptr<UVCCtrl> ctrl(UVCCtrl::create(vid, pid, location));

    if (!ctrl) {
        std::cerr << "Failed to create UVCCtrl" << std::endl;
        return 1;
    }

    // Read out initial settings
    bool oldAutoExposure;
    int oldExposure;
    int oldGain;
    ctrl->getAutoProperty(CAPPROPID_EXPOSURE, &oldAutoExposure);
    ctrl->getProperty(CAPPROPID_EXPOSURE, &oldExposure);
    ctrl->getProperty(CAPPROPID_GAIN, &oldGain);

    std::cout << "Disable autoexposure was " << static_cast<int>(!oldAutoExposure)
              << ", will set to " << disableAutoexposure << std::endl;
    std::cout << "Exposure was " << oldExposure << ", will set to " << exposure << std::endl;
    std::cout << "Gain was " << oldGain << ", will set to " << gain << std::endl;

    if (argc == 11) {
        int oldSaturation, oldHue, oldWhiteBalance;
        bool oldDisableAutoWhiteBalance;
        ctrl->getProperty(CAPPROPID_SATURATION, &oldSaturation);
        ctrl->getProperty(CAPPROPID_HUE, &oldHue);
        ctrl->getProperty(CAPPROPID_WHITEBALANCE, &oldWhiteBalance);
        ctrl->getAutoProperty(CAPPROPID_WHITEBALANCE, &oldDisableAutoWhiteBalance);
        
        std::cout << "Saturation was " << oldSaturation << ", will set to " << saturation << std::endl;
        std::cout << "Hue was " << oldHue << ", will set to " << hue << std::endl;
        std::cout << "White balance was " << oldWhiteBalance << ", will set to " << whiteBalance << std::endl;
        std::cout << "Disable auto white balance was " << static_cast<bool>(!oldDisableAutoWhiteBalance) << std::endl;
    }

    // Set new settings
    ctrl->setAutoProperty(CAPPROPID_EXPOSURE, !disableAutoexposure);
    ctrl->setProperty(CAPPROPID_EXPOSURE, exposure);
    ctrl->setProperty(CAPPROPID_GAIN, gain);
    if (argc == 11) {
        ctrl->setProperty(CAPPROPID_SATURATION, saturation);
        ctrl->setProperty(CAPPROPID_HUE, hue);
        ctrl->setProperty(CAPPROPID_WHITEBALANCE, whiteBalance);
        ctrl->setAutoProperty(CAPPROPID_WHITEBALANCE, !disableAutoWhiteBalance);
    }

    // Read back settings
    std::cout << "Reading back settings..." << std::endl;
    bool newAutoExposure;  
    int newExposure;
    int newGain;
    ctrl->getAutoProperty(CAPPROPID_EXPOSURE, &newAutoExposure);
    ctrl->getProperty(CAPPROPID_EXPOSURE, &newExposure);
    ctrl->getProperty(CAPPROPID_GAIN, &newGain);

    std::cout << "Disable autoexposure is now " << static_cast<int>(!newAutoExposure) << std::endl;
    std::cout << "Exposure is now " << newExposure << std::endl;
    std::cout << "Gain is now " << newGain << std::endl;

    if (argc == 11) {
        int newSaturation, newHue, newWhiteBalance;
        bool newDisableAutoWhiteBalance;
        ctrl->getProperty(CAPPROPID_SATURATION, &newSaturation);
        ctrl->getProperty(CAPPROPID_HUE, &newHue);
        ctrl->getProperty(CAPPROPID_WHITEBALANCE, &newWhiteBalance);
        ctrl->getAutoProperty(CAPPROPID_WHITEBALANCE, &newDisableAutoWhiteBalance);
        std::cout << "Saturation is now " << newSaturation << std::endl;
        std::cout << "Hue is now " << newHue << std::endl;
        std::cout << "White balance is now " << newWhiteBalance << std::endl;
        std::cout << "Disable auto white balance is now " << static_cast<bool>(!newDisableAutoWhiteBalance) << std::endl;
    }

    std::cout << "Done" << std::endl;
    return 0;
}
