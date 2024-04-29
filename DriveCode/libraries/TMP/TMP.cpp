//
// Created by jorda on 03/15/2024.
//
#include "TMP.h"

void TMP::init(SerialSelect SS){
    delay(20);
    printf_begin();
    printf("\r\nTFMPlus Library Example - 10SEP2021\r\n");

    delay(20);               // Give port time to initalize

    switch (SS) {
        case S0:
            tfmP.begin(&Serial);
        break;
        case S1:
            tfmP.begin(&Serial1);
        break;
        case S2:
            tfmP.begin(&Serial2);
        break;
        case S3:
            tfmP.begin(&Serial3);
        break;
    }

    printf("Soft reset: ");
    if (tfmP.sendCommand(SOFT_RESET, 0))
    {
        printf("passed.\r\n");
    }
    else tfmP.printReply();

    delay(500);  // added to allow the System Rest enough time to complete

    printf("Firmware version: ");
    if (tfmP.sendCommand(GET_FIRMWARE_VERSION, 0))
    {
        printf("%1u.", tfmP.version[0]); // print three single numbers
        printf("%1u.", tfmP.version[1]); // each separated by a dot
        printf("%1u\r\n", tfmP.version[2]);
    }
    else tfmP.printReply();
    // - - Set the data frame-rate to 20Hz - - - - - - - -
    printf("Data-Frame rate: ");
    if (tfmP.sendCommand(SET_FRAME_RATE, FRAME_20))
    {
        printf("%2uHz.\r\n", FRAME_20);
    }
    else tfmP.printReply();

    delay(500);
}

int16_t TMP::getDist() {
    if (tfmP.getData(tfDist, tfFlux, tfTemp)) // Get data from the device.
    {
        //printf("Dist:%04icm ", tfDist);   // display distance,
        //printf("Flux:%05i ", tfFlux);   // display signal strength/quality,
        //printf("Temp:%2i%s", tfTemp, "C");   // display temperature,
        //printf("\r\n");                   // end-of-line.
    }
    else                  // If the command fails...
    {
        tfmP.printFrame();  // display the error and HEX dataa
    }

    return tfDist;
}