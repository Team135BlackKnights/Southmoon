#include <libusb-1.0/libusb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <stdint.h>
#define VENDOR_ID  0x05c8
#define PRODUCT_ID 0x0a00

// UVC requests
#define UVC_SET_CUR 0x01
#define UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL 0x04
#define UVC_PU_GAIN_CONTROL 0x0F

int main(int argc, char **argv) {
    int exposure = -1, gain = -1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--exposure") && i+1 < argc) {
            exposure = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gain") && i+1 < argc) {
            gain = atoi(argv[++i]);
        }
    }

    libusb_context *ctx;
    libusb_device_handle *devh;

    libusb_init(&ctx);
    devh = libusb_open_device_with_vid_pid(ctx, VENDOR_ID, PRODUCT_ID);
    if (!devh) {
        fprintf(stderr, "Could not open device\n");
        return 1;
    }

    if (exposure >= 0) {
        uint32_t exp100us = exposure / 100; // convert µs to 100µs units
        unsigned char data[4];
        data[0] = exp100us & 0xFF;
        data[1] = (exp100us >> 8) & 0xFF;
        data[2] = (exp100us >> 16) & 0xFF;
        data[3] = (exp100us >> 24) & 0xFF;

        int r = libusb_control_transfer(devh,
            0x21, UVC_SET_CUR,
            (UVC_CT_EXPOSURE_TIME_ABSOLUTE_CONTROL << 8),
            (1 << 8) | 0, // unitID=1, interface=0 (may vary!)
            data, sizeof(data), 1000);
        printf("Set exposure result: %d\n", r);
    }

    if (gain >= 0) {
        unsigned char data[2];
        data[0] = gain & 0xFF;
        data[1] = (gain >> 8) & 0xFF;

        int r = libusb_control_transfer(devh,
            0x21, UVC_SET_CUR,
            (UVC_PU_GAIN_CONTROL << 8),
            (2 << 8) | 0, // unitID=2, interface=0 (may vary!)
            data, sizeof(data), 1000);
        printf("Set gain result: %d\n", r);
    }

    libusb_close(devh);
    libusb_exit(ctx);
    return 0;
}
