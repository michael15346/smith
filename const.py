PCAP_FILE_HEADER_LEN = 24
PCAP_PACKET_HEADER_LEN = 16


class EtherType:
    IPv4 = 0x0800
    IPv6 = 0x86dd
    VLAN = 0x8100


class IpType:
    TCP = 0x06
    UDP = 0x11
